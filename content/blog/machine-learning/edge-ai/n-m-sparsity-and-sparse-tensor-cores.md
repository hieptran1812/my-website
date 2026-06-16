---
title: "N:M sparsity and Sparse Tensor Cores: the hardware that makes sparsity pay"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why most sparsity never speeds up real hardware, how NVIDIA's 2:4 pattern and Sparse Tensor Cores turn 50 percent zeros into a guaranteed 2x matmul, and exactly how to prune, retrain, and ship it with ASP and TensorRT."
tags:
  [
    "edge-ai",
    "model-optimization",
    "sparsity",
    "pruning",
    "tensor-cores",
    "tensorrt",
    "inference",
    "efficient-ml",
    "nvidia",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-1.png"
---

A few years ago I spent the better part of a sprint pruning a ResNet-50. The literature was intoxicating: papers were reporting 80, 90, even 95 percent of weights removed with barely a dent in accuracy. I wrote the masks, I fine-tuned, I watched the sparsity climb past 70 percent on the validation set with the top-1 still within a point of the dense baseline. Then I benchmarked it on the actual GPU we were going to serve on. The pruned model ran at *exactly the same speed* as the dense one. Not 2x faster, not 1.5x faster — identical, to within measurement noise. I had spent a week deleting most of the network's weights and bought myself nothing but a smaller checkpoint file.

That experience is the single most important lesson in sparsity, and almost nobody tells you up front: **the number of zeros in your weight matrix and the speed of your matmul are two different things, and they are connected only through hardware.** A weight that is zero still occupies a lane in the dense GEMM (general matrix multiply) the GPU runs. The multiply-accumulate still happens; the chip just multiplies by zero and adds nothing. You saved the file size and threw away the latency. Unstructured sparsity — the kind where any weight can be zero, the kind the accuracy papers love — is flexible and accuracy-friendly precisely because it is unconstrained, and it is slow on commodity hardware for *exactly the same reason*: the zeros land in unpredictable places, so the hardware cannot skip them without an irregular gather that costs more than it saves.

There is a coarser option that does speed things up: structured channel pruning, where you delete whole output channels or filters. That genuinely shrinks the GEMM dimensions, so it genuinely runs faster on any hardware. But it is a blunt instrument — removing an entire channel throws away every weight in it, the useful ones with the useless, and at 50 percent the accuracy cost is real and stubborn. (I cover that lever in depth in [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up).) So we are stuck between two bad options: fine-grained sparsity that keeps accuracy but is slow, and coarse sparsity that is fast but hurts.

This post is about the middle path that NVIDIA built actual silicon to make work: **N:M sparsity**, and specifically the **2:4 pattern**. The idea is almost embarrassingly simple — in every contiguous group of four weights, keep at most two nonzero — and the payoff is a *guaranteed* roughly 2x on the matmul, delivered by a dedicated hardware unit called the Sparse Tensor Core that has shipped in every NVIDIA datacenter GPU since Ampere (A100, 2020) and lives in the Jetson Orin you can put on a robot. Figure 1 shows the pattern itself: a small weight block where each group of four has its two survivors and its two pruned-to-zero positions on a fixed schedule. By the end of this post you will understand precisely *why* 2:4 is fine-grained enough to keep accuracy where channel pruning loses it, *why* the fixed pattern is what makes hardware acceleration possible where arbitrary sparsity isn't, how to derive the 2x speedup from first principles, how to prune-and-retrain a model into 2:4 with NVIDIA's ASP and PyTorch's semi-structured sparse tensors, how to stack it with int8 for a compounded win, and — just as importantly — when 2:4 is *not* worth it, because most edge accelerators do not have a Sparse Tensor Core and on them 2:4 buys you a smaller file and nothing more.

This is the sparsity lever in our four-lever frame — quantization, pruning/sparsity, distillation, efficient architecture — and it is the one place where the lever and the hardware are co-designed so tightly that the technique only exists because the silicon does. If you want the whole map of how the levers compose, start with [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression). Here we go deep on one square of it.

![A four by four grid of weights split into two groups of four where exactly two positions in each group are kept nonzero and two are pruned to zero, illustrating the fixed 2 of 4 sparsity pattern](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-1.png)

## The two kinds of sparsity, and why only one of them is fast

Before we can appreciate what 2:4 buys, we have to be precise about the thing it is fixing. "Sparsity" is one word for two completely different engineering situations, and conflating them is how people waste sprints.

**Unstructured sparsity** means any individual weight in a matrix can be zero, with no constraint on the arrangement. You compute an importance score for every weight — magnitude is the classic choice, but second-order scores like those in SparseGPT or activation-aware scores like Wanda do better for LLMs — and you zero out the bottom fraction. This is the regime where the famous accuracy results live. You really can push a well-trained CNN to 80 or 90 percent unstructured sparsity and recover most of the accuracy with fine-tuning, because the network has enormous redundancy and you are removing exactly the least useful weights, one at a time, wherever they happen to be.

The problem is that "wherever they happen to be" is poison for hardware. A modern GPU does matrix multiply by streaming dense tiles of the weight matrix and the activation matrix through Tensor Cores, multiply-accumulate units that chew through a fixed-shape block (say 16x16x16) every few cycles. The hardware has no idea your weight is zero. It loads the tile, it runs the MACs, it accumulates. A zero weight contributes `0 * activation = 0` to the sum — correct, but it still consumed a MAC slot and a cycle. To actually *skip* an unstructured zero, the hardware would have to look at each weight, decide whether to load the matching activation, and gather only the surviving activation-weight pairs into the MAC array. That gather is irregular: the surviving positions differ from row to row and column to column, so you get scattered, unpredictable memory accesses and control divergence. On a GPU, irregular gather is so expensive that for sparsity below roughly 95 to 99 percent it is *slower* than just doing the dense multiply and eating the zeros. This is not a software bug you can optimize away; it is the fundamental mismatch between an unconstrained access pattern and a machine built for regular, coalesced, tiled access. Unstructured sparsity speeds up almost nothing on commodity hardware, and that is the honest reason it rarely ships for inference acceleration even though it dominates the accuracy literature.

**Structured sparsity** means the zeros are forced into a regular, hardware-friendly arrangement. The coarsest form is channel or filter pruning: delete entire output channels of a convolution or entire rows/columns of a linear layer. Now the matmul is *literally smaller* — fewer columns in the weight matrix means a smaller, still-dense GEMM — so it runs faster on any hardware with no special support. But you have paid for that regularity with granularity. When you delete a whole channel you delete all of its weights at once, including the genuinely important ones tangled in with the dead ones. At modest sparsity that is fine; at 50 percent it bites, because you are now forced to throw away half of *every* layer's output capacity, and the accuracy recovery is harder and the ceiling lower than fine-grained pruning at the same ratio.

So the two axes are **granularity** (how fine can the zeros be) and **hardware-friendliness** (can the chip skip them). Unstructured is maximally fine and minimally friendly. Channel pruning is maximally friendly and minimally fine. N:M sparsity is the deliberate engineering compromise: fine enough to keep accuracy, regular enough to accelerate.

### The N:M idea

N:M sparsity says: partition every row of the weight matrix into contiguous groups of M consecutive elements, and within each group keep at most N nonzero. The arrangement *within* a group is free — any N of the M positions can survive — but the *constraint* is uniform across the whole matrix: every single group obeys the same N-of-M budget. The most important instance, the one NVIDIA built hardware for, is **2:4**: groups of four, keep two. That is 50 percent sparsity, applied at the granularity of individual weights, with a structure regular enough that a small fixed-size index can describe each group.

The genius of 2:4 is that it is fine-grained where it matters and structured where the hardware needs it. It is fine-grained because the choice of *which* two weights to keep is made independently per group of four — so you can keep the two important weights in one group and the two important (but differently positioned) weights in the next, almost like unstructured pruning at the local scale. But it is structured because the *count* is fixed: exactly (at most) two survivors per four, everywhere, forever. That fixed count is the whole ballgame for hardware, as we will see — it means the compressed representation has a constant, predictable size, and the skip logic is a small fixed-width selection rather than an open-ended gather.

#### Worked example: counting the survivors

Take a single linear layer with a weight matrix of shape 4096 x 4096 — about 16.8 million weights, a typical feed-forward projection in a mid-size transformer. Under 2:4, we tile each row of 4096 into 1024 groups of four. In each group we keep two, so we keep 2048 of the 4096 per row, and 4096 x 2048 = 8.4 million weights survive across the layer. Exactly half are zero, *guaranteed*, with no group ever keeping three or four. Compare unstructured 50 percent pruning of the same layer: you would also keep 8.4 million weights, but they could be distributed any way at all — one row might keep 90 percent of its weights and another 10 percent. The 2:4 version is more constrained (every group keeps exactly two), and it is that constraint that the silicon exploits. We have not lost much representational freedom — within each group of four we still pick the best two — but we have gained a structure the hardware can read in a single fixed-width step.

## The science: how the Sparse Tensor Core gets 2x

Now the fun part — deriving the speedup from the actual mechanics. NVIDIA's Sparse Tensor Cores arrived with the Ampere architecture (A100, GA10x, 2020) and continue in Ada (RTX 40xx), Hopper (H100), Blackwell, and the Jetson Orin SoCs that put Ampere-class GPUs on edge boards. They accelerate one specific thing: a matmul where one operand (the weights) is in 2:4 structured-sparse format. Let me build up why this gives 2x and not less, and what eats into the ideal.

### The compressed format

A dense weight matrix in 2:4 form is *stored* compressed, not as a full matrix with zeros written out. For every group of four weights you store two things:

1. The **two surviving values** (the nonzeros).
2. A small **metadata index** that records *which* two of the four original positions those values came from.

How big is the index? There are $\binom{4}{2} = 6$ ways to choose which two of four positions survive, and you need to identify each survivor's original column within the group. NVIDIA's encoding uses **2 bits per surviving element** — enough to point at one of the four positions — so 2 elements x 2 bits = **4 bits of metadata per group of four**. Figure 2 lays out the layout: the two values plus that tiny per-group index, fed straight into the core with no separate gather kernel.

![A vertical stack showing how a 2:4 group is stored as two surviving values plus a 2-bit-per-element index, cutting weight storage to roughly half of dense and feeding the Sparse Tensor Core directly](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-2.png)

Let us account for the bytes concretely. In FP16 a group of four dense weights is 4 x 2 bytes = 8 bytes. In compressed 2:4: two values at 2 bytes each = 4 bytes, plus 4 bits of metadata = 0.5 bytes, for **4.5 bytes per group**. That is a 1.78x reduction in weight bytes (8 / 4.5), close to but not exactly the naive 2x because the metadata is not free. For INT8 weights the values are 4 bytes for the dense four and 2 bytes for the two survivors, plus the same 0.5 bytes metadata = 2.5 bytes, a 1.6x reduction. The metadata overhead is a larger *fraction* at lower precision because the values shrank but the index did not — a small but real detail when you are budgeting on-chip memory.

### Deriving the 2x

Here is the throughput argument, made carefully. Consider a single output element of the matmul $C = A \times B$, where $B$ is the weight matrix in 2:4 form. To compute one output we take a dot product along the contraction dimension $K$:

$$C_{ij} = \sum_{k=1}^{K} A_{ik} \, B_{kj}.$$

In the dense case this is $K$ multiply-accumulates. Now suppose $B$'s column $j$ is 2:4 sparse along $k$: in every group of four consecutive $k$ values, two of the $B_{kj}$ are zero. A zero weight contributes nothing to the sum, so the dot product is *mathematically* over only $K/2$ nonzero terms:

$$C_{ij} = \sum_{k \,:\, B_{kj} \ne 0} A_{ik} \, B_{kj}, \qquad |\{k : B_{kj} \ne 0\}| = \frac{K}{2}.$$

A dense Tensor Core still does all $K$ MACs (it does not know the zeros are there). The Sparse Tensor Core uses the 2-bit metadata to **select**, from each group of four activations $A_{ik}$, the two that line up with surviving weights, and feeds only those two into the MAC array. So it performs $K/2$ MACs to produce the same output. Same result, half the multiply-accumulates. If the MAC array is the bottleneck — which it is in a compute-bound GEMM — half the MACs means **2x the throughput**. That is the entire derivation: the speedup equals the reciprocal of the kept fraction, $1 / (N/M) = M/N = 4/2 = 2$ for 2:4. For 1:4 (keep one of four, 75 percent sparse) the ceiling would be 4x, but 1:4 is far harder on accuracy and NVIDIA's hardware path is built and tuned for 2:4.

Figure 3 contrasts the two paths directly: the dense core grinding all sixteen MACs in a four-by-four tile group versus the sparse core reading the index, selecting two activations per group, and running eight.

![A before and after comparison showing a dense Tensor Core running all multiply-accumulates versus a Sparse Tensor Core reading the index, selecting the matching activations, and running half the multiply-accumulates for about double the throughput](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-3.png)

### What eats into the ideal 2x

The 2x is a *math-throughput* ceiling. Whether you actually see it depends on whether the MACs were your bottleneck in the first place — which is the entire point of the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and the single most common reason people are disappointed by sparsity. Three things erode the headline number:

- **Index decode and activation selection.** The core must read 4 bits of metadata per group and route activations through a small multiplexer before the MACs. This is cheap — NVIDIA designed the datapath so it overlaps with compute — but it is not free, which is why measured GEMM speedups land closer to 1.5–1.8x than a clean 2.0x even on ideal shapes.
- **Memory-bound layers.** If a layer's arithmetic intensity is low — few FLOPs per byte moved, as in the skinny matmuls of batch-1 LLM decode — then the GEMM was waiting on memory, not on MACs. Halving the MACs does nothing for a layer that was never MAC-bound. You *do* still win on weight bytes (the compressed format moves ~half the weight data), which can help a memory-bound layer somewhat, but you will not see 2x. Sparsity pays best where the layer is compute-bound: large batch, large GEMM, prefill rather than decode.
- **Small or oddly shaped GEMMs.** The 2x assumes the GEMM is large enough to amortize launch and tiling overhead and that $K$ is a clean multiple of the group structure. Tiny layers, depthwise convolutions, and irregular shapes see less.

The honest summary: 2:4 gives a *real, guaranteed* speedup on compute-bound matmuls on Sparse-Tensor-Core hardware, typically 1.3–1.8x end-to-end on a model after the memory-bound and small-layer parts are weighed in, with the ideal GEMM kernel hitting close to 2x. That is dramatically better than unstructured sparsity (which gives roughly nothing) and it comes with a much smaller accuracy cost than channel pruning at the same 50 percent.

### Why a *fixed* pattern is what makes this possible

It is worth dwelling on why 2:4 works in hardware where unstructured 50 percent does not, because the difference is subtle and it is the whole reason NVIDIA chose this exact constraint. Three properties of the fixed N:M pattern make it silicon-friendly:

1. **Constant compression ratio.** Because *every* group keeps exactly N of M, the compressed matrix has a *fixed, predictable size* — you know in advance that column $j$ has $K/2$ survivors, laid out in a regular stride. There is no variable-length encoding, no per-row offset table, no prefix-sum to find where row $i+1$ starts. The address arithmetic is trivial and identical for every group, so it can be hardwired.
2. **Bounded, local selection.** The "gather" is not an open-ended search over the whole row; it is a fixed 2-of-4 multiplexer within a tiny window. The metadata is 2 bits per element, the selection is a constant-width operation, and it fits inside the Tensor Core's existing tile pipeline. Contrast unstructured: the survivors could be anywhere, so the gather index is wide and the access stride is data-dependent — exactly the irregular pattern GPUs are bad at.
3. **Coalesced, predictable memory access.** Because the structure is uniform, loading the compressed weights is a regular, coalesced streaming read, just like dense — no scattered loads, no divergence. The activations are selected at the last moment by the small mux, so the expensive part of memory access stays regular.

This is co-design: NVIDIA picked the *most* permissive structure (finest granularity, best accuracy) that still satisfies "constant size + bounded local selection + coalesced access," and 2:4 is that sweet spot. Make it finer (allow any count per group) and you lose constant size; make it coarser (channel pruning) and you lose accuracy. The pattern is not arbitrary — it is the maximum fineness the hardware constraint allows.

### A closer look at the datapath

It helps to trace one tile through the unit, because the design choices become obvious once you see where the bytes flow. A dense Tensor Core operation on Ampere takes a tile of the activation matrix $A$ and a tile of the weight matrix $B$, both fully populated, and produces a tile of $C$ by running a small systolic array of multiply-accumulators. The array has a fixed number of MAC lanes, and on every issue it consumes one column of $A$ and one row of $B$ and fans the products into the accumulators. Throughput is set by how many MACs the array can retire per cycle, and for a dense FP16 op the array is fully occupied — every lane multiplies a real (possibly zero) weight by a real activation.

The sparse op changes one thing: the weight operand arrives compressed, holding only the survivors, accompanied by the metadata stream. Now the array still has its full complement of MAC lanes, but each lane is fed by a *selection multiplexer* sitting in front of it. The mux reads the 2-bit index for its group and routes the matching activation from a small local window of four into the lane. Because half the weights are gone, the array can be fed *two groups' worth of survivors in the time a dense array processed one group* — that is the source of the 2x. The MAC array is the same silicon; it is simply kept busy with twice as many *useful* products per unit time because the useless (zero) ones never enter. The mux and the index decode are the added hardware, and they are small precisely because the window is four wide and the index is two bits.

This is why the speedup is a property of the *weight* operand and not the activation operand: only the weights are pre-pruned to 2:4 and carry metadata; the activations are dense and get *selected* by the mux at the last moment. It is also why the sparse op has dtype and shape rules — the mux and the array are built for specific element widths (FP16, BF16, INT8, and on newer parts FP8 and INT4), and the tile dimensions must align with the array geometry. Hand the unit a shape it cannot tile cleanly and the driver falls back to a dense kernel, which is correct but gives no speedup. That fallback is invisible unless you read the kernel selection log, and it is the number-one reason a "2:4 model" shows a 1.0x speedup in practice.

One more consequence worth internalizing: the sparse op does not reduce the *number of accumulator updates per output* below what the math requires — it reduces the number of *issued MACs* by skipping the zero-weight ones. So the arithmetic is identical to a dense matmul that happened to multiply by zero in those lanes; the result is bit-for-bit what you would get by materializing the zeros and running dense (modulo the usual floating-point summation-order caveats). There is no approximation in the inference path — the only approximation is upstream, in the *choice* of which weights to zero during pruning. The hardware is exact; the modeling decision is where accuracy is spent.

## Why 50 percent fine-grained beats 50 percent coarse on accuracy

We have argued that 2:4 keeps accuracy better than channel pruning at the same 50 percent ratio. Let me make that rigorous rather than asserted, because it is the reason 2:4 is worth the trouble.

Think about what each scheme is *allowed* to remove. Channel pruning operates on entire output channels — it must remove a whole column of the weight matrix or none of it. So its decision variable is per-channel: keep channel or kill channel. With $C$ channels and a 50 percent budget it picks the best $C/2$ channels. But every weight in a killed channel dies, even the large, important ones, and every weight in a kept channel lives, even the near-zero ones. The granularity of the decision is the channel, so the *resolution* at which it can match the network's actual redundancy is coarse.

2:4 operates per group of four weights and keeps the best two in *each* group. Its decision variable is far finer: for each of the millions of 4-tuples it independently keeps the two largest-magnitude (or otherwise most-important) weights. Formally, magnitude-based 2:4 solves, group by group, the problem "keep the two weights that minimize the reconstruction error of this group's contribution" — and because the choice is local and unconstrained within the group, it can keep an important weight in position 1 of one group and an important weight in position 4 of the next. Channel pruning cannot do that; it is forced to keep or kill positionally aligned across all rows.

The quantitative intuition: the error you introduce by pruning is, to first order, the sum of squared magnitudes of the weights you removed (for magnitude pruning) or a Hessian-weighted version of it (for second-order methods). A scheme that can remove *the smallest* weights everywhere removes less total magnitude than a scheme forced to remove *whole groups* of weights regardless of their individual sizes. Fine-grained 2:4 removes the local bottom-half by magnitude in each group; channel pruning removes whole channels whose *average* importance is low but which still contain individually large weights. So for the same 50 percent budget, 2:4's removed-magnitude is smaller, its first-order error is smaller, and after fine-tuning its accuracy recovery is better. This is why, empirically, 2:4 with retraining lands within a few tenths of a point of dense on ImageNet classifiers and within a point on many transformers, while 50 percent channel pruning of the same models typically costs one to several points before you fight it back with more training.

Let me make this concrete with a toy that captures the mechanism. Suppose the weights in a layer are independent draws from a standard normal $w \sim \mathcal{N}(0, 1)$, and pruning error is measured as the sum of squared removed weights, $E = \sum_{k \in \text{removed}} w_k^2$. Under unstructured 50 percent pruning you remove the global smallest-magnitude half; the removed mass is the lower half of the half-normal distribution of $|w|$, which works out to a small fraction of the total squared magnitude — roughly 9 percent of $\sum w^2$ for a normal, because the smallest weights carry little energy. Under 2:4 you remove, in each independent group of four, the *two* smallest of four draws; because the choice is local you cannot always catch the global smallest, but with only four candidates the local-bottom-two and the global-bottom-half overlap heavily, so the removed mass is only modestly higher — roughly 11–13 percent of $\sum w^2$ in simulation. Under channel pruning you must remove whole columns: you keep the $C/2$ columns with the largest *aggregate* magnitude and drop the rest, which means you are forced to discard every weight in a dropped column including its large ones — the removed mass jumps to a substantially larger fraction because column-level selection cannot protect individual large weights. The ordering $E_{\text{unstructured}} < E_{\text{2:4}} \ll E_{\text{channel}}$ is exactly the empirical accuracy ordering, and the small gap between unstructured and 2:4 (versus the large gap to channel) is *why* 2:4 captures almost all of unstructured's accuracy while channel pruning gives most of it up. The fineness of the group is doing the work: at group size four, the local decision is almost as good as the global one.

There is a second-order subtlety the toy hides. The real importance of a weight is not its magnitude alone but its magnitude weighted by how sensitive the loss is to it — formally the diagonal of the Hessian, which is what SparseGPT and the optimal-brain-surgeon family estimate. For most well-trained CNNs the magnitude proxy is good enough at group size four that retraining closes the gap. For LLMs, where activations have extreme outlier features and a single mispruned weight in a sensitive projection can shift many downstream tokens, the Hessian-aware choice matters more, which is exactly why the LLM 2:4 results lean on SparseGPT rather than plain magnitude. The principle is the same — pick the two of four that cost the least — but the cost function gets more careful as the stakes rise.

Figure 4 puts the three schemes side by side on the axes that matter: granularity, accuracy hit, hardware speedup, and memory-access regularity. It is the clearest one-glance summary of why 2:4 is the only square that is *both* fine-grained and accelerated.

![A comparison matrix of unstructured fifty percent, channel pruning fifty percent, and 2:4 structured sparsity across granularity, accuracy hit, hardware speedup, and memory access regularity, showing 2:4 as the only scheme that is both fine-grained and hardware accelerated](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-4.png)

The reason this matters for the four-lever frame is that sparsity is the lever where the *hardware* decides whether the lever moves at all. Quantization (int8, int4) speeds up almost any modern accelerator. But sparsity only pays on hardware built for the specific structure you chose, and 2:4's structure pays on exactly one family: NVIDIA Sparse Tensor Cores. Choose your sparsity to match your silicon, or do not bother.

## The training recipe: ASP, prune then retrain

You do not get a good 2:4 model by training dense and zeroing weights at inference time. You will lose accuracy, because the surviving weights were optimized to work *alongside* the ones you just deleted. The standard recipe is **ASP — Automatic SParsity** — NVIDIA's reference workflow, and it is refreshingly simple: train dense, apply the 2:4 mask once, then fine-tune the surviving weights with the mask held fixed so they re-adapt to the absence of their pruned partners. Figure 5 shows the lifecycle as a timeline.

![A six step timeline of the ASP recipe showing train dense, score magnitudes per group of four, stamp the 2:4 mask, retrain the survivors with the mask fixed, recover dense accuracy, then export a sparse TensorRT engine](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-5.png)

The steps, with the reasoning:

1. **Train (or start from) a dense model.** You need a good dense baseline first. ASP is a finishing step, not a from-scratch training method. If you have a pretrained checkpoint, use it.
2. **Compute the 2:4 mask.** For each contiguous group of four weights along the contraction dimension, keep the two with the largest magnitude (or a smarter importance score) and mark the other two for zeroing. This is a one-shot, deterministic operation — no search, no iteration. ASP picks the dimension and grouping that match the Tensor Core's expectations so the resulting sparsity is *the* 2:4 the hardware accelerates.
3. **Apply the mask** — zero the pruned positions.
4. **Fine-tune with the mask fixed.** Retrain on the original task (often just a fraction of the original schedule — sometimes the full schedule for best results, sometimes 1–10 percent of it for a quick recovery) while *keeping the pruned positions clamped to zero*. The gradient updates only the survivors; the mask never changes. This is the step that recovers accuracy: the survivors learn to compensate for the missing weights. Mathematically the optimization is the same as normal training but constrained to the subspace where the masked weights stay zero — you are fine-tuning on the sparse manifold.
5. **Validate** that accuracy is back to ~dense. For well-trained CNNs and many transformers this recovery is reliable; the published result from Mishra et al. (NVIDIA, "Accelerating Sparse Deep Neural Networks", 2021) is that across a broad sweep of vision and language models, 2:4 with retraining matched the dense baseline accuracy.
6. **Export the sparse engine.** Convert to a deployable format that the runtime recognizes as 2:4 — in practice a TensorRT engine built with sparsity enabled, or a PyTorch semi-structured sparse tensor dispatched to the cuSPARSELt-backed kernels.

The reason a *fixed mask during fine-tuning* matters: if you let the mask move (re-pruning every step based on current magnitudes), you get a form of dynamic sparse training that can reach slightly higher accuracy but is finicky and the mask must still resolve to a valid 2:4 layout at export. The ASP default — one-shot mask, fixed during retrain — is the boring, reliable choice that ships. There is also a "mask the gradients too" subtlety: you want the optimizer's momentum and weight-decay state for the dead positions to stay zero, or they will drift; ASP handles this by wrapping the optimizer.

### Why one-shot magnitude pruning is good enough here

You might expect that picking which two of four to keep needs a clever importance metric. For 2:4 it usually does not, and the reason is the small group size. With only four candidates and a budget of two, the magnitude ordering and the optimal (Hessian-aware) ordering agree most of the time — there is little room for the cleverness to matter when you are choosing two of four. This is why simple magnitude-based ASP recovers dense accuracy on most models: the granularity is fine enough that the *which-two* decision is rarely wrong in a way fine-tuning cannot fix. For LLMs, where even a small per-layer error can compound across many layers, second-order one-shot methods (SparseGPT applied to a 2:4 budget) do measurably better than magnitude and skip the retraining entirely, which matters when retraining a 70B model is infeasible.

## The practical flow: enforcing 2:4 and measuring the win

Enough theory. Here is how you actually do it in code, end to end, with the two mainstream paths: NVIDIA's ASP library and PyTorch's built-in semi-structured sparse tensors. Both ultimately dispatch to the same family of cuSPARSELt / Sparse-Tensor-Core kernels.

### Path A: NVIDIA ASP (apex.contrib.sparsity)

ASP lives in NVIDIA's Apex repo under `apex.contrib.sparsity`. It wraps your model and optimizer, computes the 2:4 masks, and keeps them enforced through fine-tuning. The whole integration is a handful of lines around your existing training loop.

```python
import torch
import torchvision
from apex.contrib.sparsity import ASP

# 1. Start from a good dense checkpoint
model = torchvision.models.resnet50(weights="IMAGENET1K_V2").cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                            weight_decay=1e-4)

# 2. Tell ASP to prune the model to 2:4 and own the optimizer state
#    init_model_for_pruning computes the masks (one-shot magnitude by default);
#    init_optimizer_for_pruning keeps masked weights (and their momentum) at zero.
ASP.prune_trained_model(model, optimizer)
#   ^ convenience call = init_model_for_pruning + compute_sparse_masks
#     + init_optimizer_for_pruning, all in one.

# 3. Fine-tune exactly as normal — the mask is enforced automatically every step.
for epoch in range(num_finetune_epochs):
    for images, targets in train_loader:
        images, targets = images.cuda(), targets.cuda()
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()        # masked positions stay zero
    validate(model, val_loader)  # watch top-1 climb back to dense

# 4. Save. The state_dict now holds a 2:4-sparse model.
torch.save(model.state_dict(), "resnet50_2to4.pth")
```

The important detail is step 2: `prune_trained_model` both stamps the masks and rewrites the optimizer so that the gradient, the momentum buffer, and weight decay for the dead positions are forced to zero. Without that, weight decay alone would slowly pull masked weights away from zero and the layout would become invalid at export. By default ASP prunes the layers it knows the Tensor Core can accelerate (linear and conv weights along the right dimension) and leaves the rest dense — you do not 2:4 a bias or a 1x1 with the wrong shape.

### Path B: PyTorch semi-structured sparse

Since PyTorch 2.1, the framework has first-class support for 2:4 (it calls it "semi-structured sparsity") via `torch.sparse`. You enforce the 2:4 mask yourself (or with `torch.ao.pruning`'s `WeightNormSparsifier` configured for 2:4), then convert the dense-but-masked tensor into the compressed semi-structured layout that dispatches to the accelerated kernel.

```python
import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# Opt in to the cuSPARSELt backend for best kernel coverage.
SparseSemiStructuredTensor._FORCE_CUTLASS = False  # use cuSPARSELt if available

# Suppose `weight` already satisfies the 2:4 mask (e.g. produced by ASP-style
# pruning + fine-tune above). Shape and dtype must be Sparse-TC compatible:
# fp16/bf16/int8, with the contraction dim a multiple of the group size.
weight = weight.half().cuda().contiguous()

# Convert to the compressed 2:4 representation (values + metadata).
sparse_weight = to_sparse_semi_structured(weight)

# The sparse tensor multiplies like a dense one but dispatches to the
# Sparse Tensor Core kernel. Note the operand orientation: the SPARSE
# operand is the one whose zeros get skipped.
activations = torch.randn(4096, 8192, dtype=torch.half, device="cuda")
out = torch.mm(sparse_weight, activations)   # accelerated 2:4 matmul

# Drop-in for an nn.Linear: swap the weight for its semi-structured form.
import torch.nn as nn
def sparsify_linear(linear: nn.Linear) -> nn.Linear:
    assert linear.weight.dtype in (torch.float16, torch.bfloat16, torch.int8)
    linear.weight = nn.Parameter(
        to_sparse_semi_structured(linear.weight.data.contiguous())
    )
    return linear
```

A subtlety that trips people: which operand is sparse. In `torch.mm(A, B)` the Sparse Tensor Core skips zeros in the *sparse* operand. For an `nn.Linear`, the weight is the natural sparse operand, but the kernel has orientation constraints (the sparse operand is typically the first/left matrix in the accelerated path), so the framework may transpose it internally. If you see no speedup, check that the sparse tensor is actually the operand being skipped and that the shapes are Tensor-Core-aligned (dimensions multiples of 8 or 16 depending on dtype). This is also where a flat dense fallback silently happens: if the shape or dtype is not supported, PyTorch will run a dense matmul and you get correctness but no speed — exactly the trap from my ResNet story, in a new costume.

### Measuring the speedup honestly

You must benchmark on a Sparse-Tensor-Core GPU (Ampere or newer: A100, A10, A30, RTX 30xx/40xx, H100, Jetson Orin) and you must measure correctly or you will fool yourself. The rules: warm up the kernels (the first call pays compilation and allocation costs), synchronize the GPU before and after timing (CUDA is asynchronous — a naive wall-clock measures launch, not execution), run many iterations and report the median, and pin the clocks if you care about reproducibility (thermal throttling on a sustained run will quietly lower your numbers; lock with `nvidia-smi -lgc`).

```python
import torch, time
from torch.sparse import to_sparse_semi_structured

def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms per call

K, N, M = 8192, 8192, 8192
dense_w = torch.randn(K, N, dtype=torch.half, device="cuda")
# Stamp a valid 2:4 mask: keep top-2 magnitude in each group of 4 along dim 1.
g = dense_w.abs().reshape(K, -1, 4)
keep = g >= g.topk(2, dim=-1).values[..., -1:].expand_as(g)
masked_w = (dense_w.reshape(K, -1, 4) * keep).reshape(K, N).contiguous()

sparse_w = to_sparse_semi_structured(masked_w)
x = torch.randn(N, M, dtype=torch.half, device="cuda")

dense_ms  = bench(lambda: torch.mm(masked_w, x))
sparse_ms = bench(lambda: torch.mm(sparse_w, x))
print(f"dense  {dense_ms:.3f} ms")
print(f"sparse {sparse_ms:.3f} ms   speedup {dense_ms / sparse_ms:.2f}x")
```

On a large, square, FP16 GEMM like this an A100 or H100 will typically report a sparse/dense ratio in the **1.5–1.8x** range — the index-decode and selection overhead is what separates that from a clean 2.0x. Shrink the GEMM, make it memory-bound, or run it on a small Orin and the ratio drops toward 1.0x; that is not a bug, it is the roofline reminding you that sparsity only helps where MACs were the bottleneck.

### TensorRT sparse build

For production inference on NVIDIA hardware the path that actually ships is TensorRT. If your weights are already 2:4 (from ASP), you build the engine with sparsity enabled and TensorRT will select sparse kernels wherever they are faster than the dense alternative — it benchmarks both during the build and picks the winner per layer, so you never get *slower* by turning it on.

```bash
# Build a TensorRT engine that is allowed to use Sparse Tensor Core kernels.
# --sparsity=enable assumes the weights already satisfy 2:4.
# --sparsity=force will prune-to-2:4 internally (use only if you accept the
#   accuracy hit of unretrained one-shot pruning).
trtexec \
  --onnx=resnet50_2to4.onnx \
  --sparsity=enable \
  --fp16 \
  --saveEngine=resnet50_2to4.plan \
  --noDataTransfers --useCudaGraph \
  --avgRuns=200 --warmUp=2000

# Compare against the dense engine to confirm the win is real.
trtexec --onnx=resnet50_dense.onnx --fp16 \
        --saveEngine=resnet50_dense.plan --avgRuns=200 --warmUp=2000
```

The builder log will tell you, per layer, whether it chose a sparse or dense tactic. Read it: if every layer fell back to dense, your weights were not actually in valid 2:4 layout (a common cause is pruning along the wrong dimension), or the layers were too small/memory-bound for the sparse kernel to win. The `--sparsity=force` flag is a footgun for accuracy — it prunes to 2:4 *without retraining*, so use it only to estimate the speedup ceiling, never for a model you ship.

## 2:4 on large language models without retraining

CNNs are the easy case for 2:4 because retraining them is cheap. LLMs are the hard and interesting case, because the dense model cost millions of dollars to train and *re*training it to recover from pruning is usually off the table. The whole game for LLM 2:4 is therefore to make the one-shot prune so accurate that no recovery training is needed — and that is exactly what SparseGPT and Wanda are built for.

The reason a careless one-shot prune wrecks an LLM is worth understanding. In a transformer, the feed-forward and attention projections are large linear layers, and a small error in one layer's output becomes the input to the next, so per-layer errors *compound* across dozens of layers. Plain magnitude pruning ignores this: it zeros the smallest-magnitude weights without asking how much the *loss* moves when you do. SparseGPT instead solves, layer by layer, an approximation to "remove half the weights (in 2:4 layout) so that this layer's output on a calibration set changes as little as possible." It uses the inverse Hessian of the layer's reconstruction objective — estimated cheaply from a few hundred calibration sequences — to decide which two of four to keep *and* to adjust the surviving weights to compensate for the removed ones, all in closed form. That compensation step is the magic: it is a one-shot, training-free analog of the ASP fine-tune, baked into the pruning math. Wanda is the budget version — it scores weights by magnitude times the norm of the corresponding input activation (an activation-aware proxy for sensitivity), which captures most of SparseGPT's benefit at a fraction of the cost and with no weight update.

```python
# Pruning a HuggingFace LLM to 2:4 with a SparseGPT-style one-shot pass.
# (Conceptual flow; libraries like NVIDIA's nvidia-modelopt or community
#  SparseGPT repos implement the Hessian math.)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 1. Gather a small calibration set (a few hundred sequences is enough).
calib = [tok(s, return_tensors="pt").input_ids.cuda()
         for s in load_calibration_texts(n=256)]

# 2. For each linear layer, capture inputs, estimate the layer Hessian
#    H = sum_x x x^T over the calibration activations, then choose a 2:4
#    mask + update survivors to minimize ||W X - W_pruned X||.
prune_to_2to4_sparsegpt(model, calib, blocksize=4)   # the heavy lifting

# 3. Export to a 2:4-aware engine (TensorRT-LLM / cuSPARSELt path).
model.save_pretrained("llama2-7b-2to4")
```

The measured story for LLM 2:4 is honest but more nuanced than the CNN story. SparseGPT can prune a 7B–70B model to 2:4 in a few GPU-hours and the perplexity increase is far smaller than magnitude pruning — but it is *not* free the way ResNet 2:4 is. Expect a perplexity bump of a fraction of a point to a couple of points depending on model and budget, larger on smaller models (which have less redundancy to spare) and smaller on the big ones. And remember the roofline: LLM *decode* is memory-bound (one token at a time, each weight read once), so the 2:4 MAC-skip helps decode little — the win there is the halved weight bytes you stream, similar to what int4 weight-only quantization gives. The 2:4 compute win shows up at *prefill* and at large serving batch, where the GEMMs are big and compute-bound. So for a latency-sensitive single-stream chatbot, weight-only int4 (covered in [weight-only LLM quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)) is often the better lever than 2:4; for a high-throughput batched serving deployment on H100s, 2:4 (or 2:4 + FP8) pulls real weight on the prefill and batched GEMMs.

## Stacking 2:4 with int8

The single best property of 2:4 for an edge-systems engineer is that it composes cleanly with quantization. They cut *different* costs — sparsity halves the multiply-accumulates, int8 quarters the bytes and speeds up the narrow-element math path — so on Sparse-Tensor-Core hardware that also has an INT8 path (every Ampere+ GPU does), the two wins largely multiply rather than overlap. Figure 6 shows the stacking.

![A before and after comparison showing dense FP16 weights versus combined 2:4 sparse INT8 weights, where int8 makes the values four times smaller and the 2:4 pattern skips half the multiply accumulates so the gains compound](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-6.png)

The mechanics: NVIDIA's Sparse Tensor Cores accelerate 2:4 in FP16, BF16, *and* INT8 (and INT4 / FP8 on newer architectures). So you can have an INT8 weight that is *also* 2:4 sparse: each group of four int8 weights keeps two, with the 2-bit metadata, and the core skips the two zeros while running the int8 MAC path. The size math from earlier: an int8 dense group of four is 4 bytes; the 2:4 int8 group is 2 bytes of values + 0.5 bytes metadata = 2.5 bytes, so vs FP16 dense (8 bytes) the 2:4+int8 weight is **3.2x smaller**. On throughput, the int8 Tensor Core path is roughly 2x the FP16 path to begin with, and 2:4 adds its ~1.5–2x on top, so the compounded matmul speedup over dense FP16 can reach the **3–4x** range on compute-bound layers.

The order of operations matters and there is a right way to do it: **quantize and sparsify together, then fine-tune once.** If you prune to 2:4, fine-tune, then quantize and fine-tune again, you do twice the work and the second fine-tune can disturb the sparsity. The clean recipe is to apply the 2:4 mask, attach the int8 fake-quant (quantization-aware training, see [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat)), and do *one* recovery fine-tune where the network simultaneously adapts to the missing weights and the reduced precision. The survivors learn to be both sparse-robust and quantization-robust at once. If you have a strong post-training-quantization pipeline you may be able to skip the QAT and do 2:4 + PTQ, but for the combined hit it is usually worth one joint fine-tune.

```python
# Sketch: 2:4 mask + int8 QAT in one fine-tune (PyTorch).
import torch
from apex.contrib.sparsity import ASP
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

model = build_dense_model().cuda()
model.train()

# (a) stamp the 2:4 masks and own the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
ASP.prune_trained_model(model, optimizer)         # now 2:4 sparse

# (b) attach int8 fake-quant on top of the sparse model
model.qconfig = get_default_qat_qconfig("x86")    # or a TensorRT-friendly config
prepare_qat(model, inplace=True)                  # inserts fake-quant observers

# (c) ONE joint recovery fine-tune: learns sparse + int8 together
for epoch in range(joint_finetune_epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x.cuda()), y.cuda())
        loss.backward()
        optimizer.step()         # masked weights stay zero; fake-quant active

# (d) freeze to a real int8 + 2:4 model and export to ONNX -> TensorRT
model.eval()
int8_sparse = convert(model.cpu().eval())
# export to ONNX, then: trtexec --onnx=... --int8 --sparsity=enable ...
```

The one caveat when stacking: validate the *combined* accuracy, not each step in isolation. Sparsity removes capacity and quantization removes precision, and while each alone may cost only a few tenths of a point, the interaction can be slightly super-additive on a sensitive layer. The fix is the usual one — keep the most sensitive layers (often the first conv and the final classifier) dense and/or higher precision, which is the [mixed-precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) lever applied to sparsity.

## Worked examples with real numbers

#### Worked example: ResNet-50 pruned to 2:4 on an A100

Start with the standard torchvision ResNet-50, dense FP16, with a reference ImageNet top-1 around 76.1 percent (the V2 weights are higher, but use 76.1 as a clean baseline number for this example). Run ASP: one-shot magnitude 2:4 masks on the conv and linear weights, then a short fine-tune (a handful of epochs at a low learning rate on ImageNet). The published NVIDIA result for this exact setup is that 2:4 with retraining **recovers to within ~0.1–0.3 points of the dense baseline** — call it 75.9 percent, a 0.2-point drop, which is in the noise of a single training run.

Now the speed. On an A100, a well-shaped FP16 GEMM with 2:4 weights runs the matmul at roughly **1.5–1.8x** of dense; across the whole ResNet-50, where some layers are small or memory-bound and the non-GEMM parts (BN, pooling, activations) do not speed up, the *end-to-end* inference speedup is more modest, typically in the **1.3–1.5x** range at a serving batch size where the convolutions are compute-bound. Model size: the 2:4-sparse FP16 weights compress to roughly **55 percent** of the dense FP16 size (the ~1.78x weight-byte reduction, diluted by the parts you left dense). So the deal is: 0.2 points of accuracy for ~1.4x throughput and ~0.55x weight bytes, on hardware you already own if you serve on NVIDIA. That is a clean Pareto move — strictly better latency at negligible accuracy cost.

The stress test: drop the batch size to 1 and re-measure. Now many of ResNet-50's layers are memory-bound (each weight is used once per inference, so arithmetic intensity is low), and the sparse speedup shrinks toward 1.1–1.2x because you were not MAC-bound. The lesson is the roofline lesson: 2:4 pays at the batch sizes and layer shapes where you were compute-bound, and at strict batch-1 on a small model it pays much less. If your serving reality is batch-1, sparsity helps less than the headline; measure *your* shapes.

#### Worked example: a transformer FFN with 2:4 + int8 on an H100

Take the feed-forward block of a mid-size transformer — two large linear layers (the 4x expansion and the projection back), the most compute-heavy part of the model at prefill. Dense FP16, these GEMMs are squarely compute-bound at a reasonable sequence length and batch. Apply 2:4 + INT8 with one joint QAT fine-tune.

The bytes: each FFN weight goes from FP16 dense (2 bytes/weight) to int8 2:4 (effectively ~0.625 bytes/weight after the half-survivors and metadata), about a **3.2x** reduction in FFN weight storage. The throughput: the int8 Tensor Core path on H100 is roughly 2x the FP16 path, and 2:4 stacks ~1.5–1.8x on the compute-bound GEMM, for a combined matmul speedup over dense FP16 in the **3–3.5x** range on the FFN's large layers. Accuracy: with a joint fine-tune, the combined drop on a well-behaved model stays within roughly **0.5–1.0 point** of the dense FP16 baseline on the task metric — the joint recovery lets the survivors absorb both the missing partners and the coarser precision. The honest caveat from the stress test: at *decode* time (batch-1, one token at a time), the FFN GEMMs are skinny and memory-bound, so the int8 byte reduction still helps (less weight data to stream) but the 2:4 MAC-skip helps little. The big sparse+int8 win shows up at prefill and at large batch; the win at single-stream decode is dominated by the int8 byte savings, not the sparsity.

These two examples bracket the reality: **2:4 + int8 is a server/datacenter and Jetson-Orin-class win on compute-bound work, and a partial win at memory-bound batch-1 decode.** Know which regime your deployment lives in before you promise a number.

## Results: the trade-off tables

Two tables capture the decision. The first is the *progression* — dense to 2:4 to 2:4+int8 — and what each step buys, summarized in figure 7. The numbers are representative of a compute-bound ResNet-50-class workload on an Ampere/Hopper GPU; treat them as order-of-magnitude, and always re-measure on your model and your target.

![A matrix comparing dense FP16, 2:4 sparse FP16, and 2:4 plus int8 across latency, model size, top-1 accuracy, and hardware requirement, showing the speed and size wins compounding while accuracy stays within a point](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-7.png)

| Scheme | Matmul speedup (compute-bound) | Weight bytes | Top-1 vs dense | Hardware needed |
|---|---|---|---|---|
| Dense FP16 | 1.0x (baseline) | 100% | 76.1% (ref) | any GPU |
| 2:4 sparse FP16 | ~1.5–1.8x | ~55% | -0.2 pt | Ampere+ Sparse Tensor Core |
| INT8 dense | ~2x | ~25% | -0.3 pt | any int8 GPU |
| 2:4 + INT8 | ~3–4x | ~30% of FP16 | -0.5 to -1.0 pt | Sparse TC + int8 path |

The second table is the one that should change how you think about pruning at all — the same 50 percent of weights removed three different ways, and what each actually delivers on a real GPU.

| 50% removal scheme | Granularity | Accuracy hit (retrained) | Speedup on commodity GPU | Memory access |
|---|---|---|---|---|
| Unstructured 50% | finest (any weight) | smallest (~0 pt) | ~none (dense fallback) | irregular gather |
| Channel-prune 50% | coarsest (whole channels) | largest (1–3+ pt) | real (smaller dense GEMM) | contiguous |
| 2:4 structured | fine, fixed (2 of 4) | small (~0.2 pt) | ~1.5–1.8x (native) | predictable, coalesced |

Read those two tables together and the thesis of the whole post falls out: unstructured wins on accuracy but loses on speed; channel pruning wins on universal speed but loses on accuracy; **2:4 is the only row that is good on *both* axes — and it is good on both precisely because NVIDIA built the silicon to make a fine-grained-but-fixed pattern fast.** That is the deal, and it is an unusually clean one in a field full of muddy trade-offs.

## Stress-testing the decision: the ways 2:4 disappoints

A technique is only as useful as your understanding of its failure modes, so let me reason through the situations where a confident 2:4 plan goes sideways. Each of these is a real way I have seen the win evaporate.

**The silent dense fallback.** You prune, you convert, you benchmark, and the speedup is 1.0x. The most common cause is that the kernel never ran sparse. PyTorch and TensorRT both fall back to a dense kernel — *correctly*, returning the right numbers — when the operand shape, dtype, or orientation does not match what the Sparse Tensor Core path requires. The contraction dimension must be a clean multiple of the tile size; the dtype must be one the sparse unit supports; and the *sparse* operand has to be the one whose zeros you want skipped, which for an `nn.Linear` can require a transpose the framework may or may not insert for you. The fix is to instrument: in PyTorch, time the `to_sparse_semi_structured` matmul against the dense one on your exact shapes and confirm a ratio above ~1.3x before trusting it; in TensorRT, read the per-layer tactic selection in the build log and confirm sparse kernels were chosen. Never assume the speedup happened because the API accepted the call.

**The memory-bound layer that was never going to speed up.** You profiled the model, saw the matmuls dominating wall-clock, and assumed they were compute-bound. But at batch-1 with single-token decode, those matmuls read each weight exactly once and do a single MAC with it — arithmetic intensity is roughly one, far left of the roofline ridge, deeply memory-bound. Halving the *MACs* of a memory-bound layer does nothing because the MACs were idle waiting on DRAM. You still win on the ~half weight bytes you stream, which gives a partial speedup, but it is the int8/int4 *byte* reduction that helps a memory-bound layer most, not the 2:4 *MAC* skip. If your deployment is single-stream decode, profile arithmetic intensity first; if you are memory-bound, reach for weight-only quantization before sparsity.

**The accuracy cliff on a too-small or too-pruned model.** 2:4 is nearly free on a *well-trained, over-parameterized* model — exactly the regime where redundancy exists to remove. On a small model trained to its capacity, or one already aggressively pruned or distilled, there may be no slack: the 2:4 mask removes weights that were actually pulling their weight, and even a full retrain cannot recover. The signature is a recovery curve that plateaus below dense no matter how long you fine-tune. The lesson: 2:4 spends the model's redundancy, and if you have already spent it on distillation or a tight architecture, there may be nothing left for sparsity to take. Stack sparsity *early* in the compression order, while redundancy remains, not last.

**The tiny calibration set for one-shot LLM pruning.** SparseGPT's Hessian estimate comes from a calibration set. Too few or too narrow a set, and the estimated inverse Hessian is noisy, the survivor-compensation overfits the calibration distribution, and the model degrades on out-of-distribution prompts even as the calibration perplexity looks fine. A few hundred diverse sequences is the usual floor; a handful of repetitive ones is a trap. This is the same calibration-set failure that bites post-training quantization, and the fix is the same: make the calibration set representative of real traffic.

**The op the engine cannot fuse.** Sparsity changes the GEMM but not the surrounding elementwise ops (bias, activation, normalization). If the runtime cannot fuse those into the sparse GEMM the way it fused them into the dense one, you can lose some of the sparse win to extra kernel launches and memory round-trips. This is a maturity issue that improves with newer TensorRT and cuSPARSELt versions, but it is worth checking that the sparse path did not break a fusion the dense path had. Profile the *full* layer, not just the GEMM in isolation.

The throughline: **2:4 is a compute-bound, redundant-model, supported-hardware, correctly-shaped technique.** Violate any of those four conditions and the win shrinks or vanishes. Check all four before you commit a sprint.

## Case studies and real numbers from the literature

A few grounded results, cited, so the numbers above are not just my representative estimates.

**Mishra et al., "Accelerating Sparse Deep Neural Networks" (NVIDIA, 2021).** This is the foundational 2:4 paper and the source of the headline claim. Across a large sweep of networks — ResNets, image classifiers, segmentation and detection backbones, and Transformer-based language models — the authors show that 2:4 structured sparsity with the prune-and-retrain recipe **recovers the dense baseline accuracy** (within run-to-run noise) while enabling the ~2x math-throughput Sparse-Tensor-Core path. The paper is also where the compressed format (2 values + 2-bit indices per group of four) and the A100 Sparse Tensor Core are documented. If you read one source on this, read that one.

**Pool and Yu, "Channel Permutations for N:M Sparsity" (NVIDIA, NeurIPS 2021).** A refinement worth knowing: the 2:4 constraint is applied along a fixed dimension, and *which* weights end up grouped together affects how much magnitude you are forced to remove. By permuting channels before applying the mask, you can cluster the important weights so that fewer of them collide within a group of four, which lowers the pruning error and recovers a bit more accuracy at the same 50 percent — for free at inference, since the permutation is folded into the surrounding layers. It is the kind of detail that separates a good 2:4 result from a great one.

**SparseGPT (Frantar and Alistarh, 2023) at a 2:4 budget.** For LLMs, retraining is often infeasible, so one-shot pruning quality matters enormously. SparseGPT is a one-shot, second-order (Hessian-aware) pruner that can target a 2:4 budget and prune a model with tens of billions of parameters to 2:4 in a few GPU-hours *without any fine-tuning*, with a perplexity increase far smaller than one-shot magnitude pruning would give. This is the practical route to 2:4 on a large LLM: you do not retrain, you prune carefully once. Wanda (Sun et al., 2023) is a cheaper activation-aware alternative in the same spirit.

**NVIDIA TensorRT and cuSPARSELt in production.** NVIDIA's own inference stack exposes 2:4 as a build flag (`--sparsity=enable`) and the cuSPARSELt library provides the underlying sparse GEMM kernels; the published end-to-end speedups for sparse-friendly models on A100/H100 land in the ~1.3–1.5x range overall (more on the heavy GEMMs, less on the rest), consistent with the per-layer ~1.5–1.8x once you average in the non-accelerated parts. This is the honest "what you get in a real engine" number, as opposed to the kernel microbenchmark ceiling.

The thread through all of these: 2:4 is a *shipped, supported, measured* technique on NVIDIA silicon, not a research curiosity. That is the difference between it and most of the unstructured-sparsity literature.

## When 2:4 is worth it, and when it absolutely is not

Time for the decisive recommendation. Figure 8 is the decision tree; the prose below is the reasoning behind each branch.

![A decision tree for whether to use 2:4 sparsity, branching first on whether you deploy on an Ampere or newer Sparse Tensor Core GPU and then on whether you need a guaranteed two times matmul, with a branch that skips sparsity on edge NPUs and microcontrollers](/imgs/blogs/n-m-sparsity-and-sparse-tensor-cores-8.png)

**Reach for 2:4 when:**

- **You deploy on a Sparse-Tensor-Core GPU.** Ampere or newer: A100, A10, A30, the RTX 30xx/40xx desktop cards, H100/Blackwell in the datacenter, and crucially the **Jetson Orin** family for edge robotics and embedded vision. This is the hard prerequisite. No Sparse Tensor Core, no matmul speedup — full stop.
- **You want a guaranteed ~2x on compute-bound matmuls without an accuracy cliff.** Unlike int4 (which can fall off a cliff on sensitive models) or channel pruning (which costs points), 2:4 with retraining is the rare lever that is nearly free on accuracy and reliable on speed. If your model is compute-bound and you serve on the right hardware, it is close to a free 1.4–1.8x.
- **You are already building a TensorRT engine.** The marginal cost of `--sparsity=enable` is one flag and one prune-and-retrain pass, and TensorRT will only use the sparse kernel where it is actually faster, so there is no downside risk to the engine's latency.
- **You want to stack it with int8.** The combination is where the big numbers live (3–4x), and it composes cleanly with one joint fine-tune.

**Do not reach for 2:4 when:**

- **Your target has no Sparse Tensor Core — which is most of the edge.** This is the honest, load-bearing caveat of the entire post. The overwhelming majority of edge accelerators do **not** have a Sparse Tensor Core: phone NPUs (Apple Neural Engine, Qualcomm Hexagon, Google Tensor's TPU), Coral/Edge-TPU, most embedded NPUs, and every microcontroller (Cortex-M with CMSIS-NN). On those, applying 2:4 gives you a smaller checkpoint file and *zero* speedup — the runtime materializes the zeros and runs a dense kernel, exactly the trap I fell into years ago. For those targets, your levers are quantization (int8 is universally fast) and architecture/NAS, not 2:4. 2:4 is mainly a **server / datacenter / Jetson-Orin-class** win. Say this to your team before they spend a sprint on it for a phone.
- **50 percent is not enough sparsity for your goal.** 2:4 is a *fixed* 50 percent ceiling — you cannot dial it to 70 or 80 percent. The hardware accelerates 2:4 (and on newer parts 2:4 in narrower types), not arbitrary ratios. If you genuinely need to remove 80 percent of the weights, 2:4 cannot do it; you are back to unstructured (accept the no-speedup) or a different approach. 1:4 exists in research but is not the broadly accelerated, accuracy-safe path that 2:4 is.
- **Your layers are memory-bound at your batch size.** If you serve strict batch-1 decode on a small model, the GEMMs are memory-bound and the MAC-skip buys little. Profile first (the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) tells you whether you are compute- or memory-bound); if you are memory-bound, prefer int8/int4 weight compression (which reduces the bytes you are bottlenecked on) over 2:4.
- **You cannot afford the retrain and cannot do a good one-shot prune.** For small models the retrain is cheap. For a giant LLM where retraining is off the table, you need a quality one-shot pruner (SparseGPT) to hit 2:4 without an accuracy cliff; plain magnitude one-shot on an LLM will hurt.

The meta-rule: **2:4 is a hardware-conditional lever.** Quantization moves on almost any accelerator; 2:4 moves only on Sparse Tensor Cores. Check your deployment hardware *first*, before you write a single mask. If the answer is "Apple Neural Engine" or "Cortex-M," close this tab and go quantize. If the answer is "A100" or "Jetson Orin," 2:4 is one of the cleanest wins you have.

## How this fits the four-lever frame

Stepping back: in the [taxonomy of compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), the four levers are quantization, pruning/sparsity, distillation, and efficient architecture, all sitting on compilers/runtimes and validated by profiling. 2:4 is a specific, hardware-co-designed point on the *pruning/sparsity* lever — the one place where the lever's effectiveness is entirely gated by whether the target silicon implements the structure. It stacks cleanly with the *quantization* lever (2:4 + int8), it is orthogonal to *distillation* (you can distill into a model and then 2:4-prune it, or 2:4-prune a teacher), and it interacts with *efficient architecture* mainly through layer shapes (it pays on the big compute-bound GEMMs, which efficient architectures sometimes deliberately shrink away). And it lives or dies by *profiling*: the roofline tells you whether the MAC-skip will help, and the only honest way to know your speedup is to measure it on your target with proper warm-up and synchronization.

When you assemble a full edge deployment, 2:4 is a tool you pull *if and only if* your hardware has the Sparse Tensor Core and your hot layers are compute-bound — and then you stack it under int8 for the compounded win. That full assembly, choosing and ordering every lever for a real target, is the subject of the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook). And which target you have in the first place — phone NPU vs Jetson vs datacenter GPU vs microcontroller — is exactly the [edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) that decides whether 2:4 is even on the table.

## Key takeaways

- **Zeros are not speed.** Unstructured sparsity shrinks the file but not the matmul on commodity hardware, because the GPU still runs the MACs and the irregular gather to skip zeros costs more than it saves below ~95 percent sparsity.
- **2:4 is the engineered middle path.** Keep two of every four weights — fine-grained enough to preserve accuracy, fixed enough that a tiny per-group index makes the hardware skip the zeros. It is 50 percent sparsity that actually accelerates.
- **The 2x comes from skipping half the MACs.** The Sparse Tensor Core reads 2 bits of metadata per element, selects the two surviving activations per group of four, and runs $K/2$ multiply-accumulates instead of $K$. Index decode pulls the measured kernel speedup to ~1.5–1.8x, not a clean 2.0x.
- **Fixed structure is the whole trick.** A constant 2-of-4 count gives a constant compression size, a bounded local selection, and coalesced memory access — the three things a GPU needs and that arbitrary sparsity cannot provide.
- **Fine-grained beats coarse on accuracy at the same ratio.** 2:4 removes the local smallest weights everywhere; channel pruning is forced to kill whole channels, removing more total magnitude, so 2:4 keeps accuracy within ~0.2 points where 50 percent channel pruning costs 1–3+.
- **The recipe is train-dense, mask-once, retrain-survivors (ASP).** Hold the mask fixed during fine-tuning so the survivors re-adapt; for LLMs that cannot retrain, use a one-shot second-order pruner (SparseGPT) at a 2:4 budget.
- **Stack it with int8 for the big win.** Sparsity halves MACs, int8 quarters bytes; they cut different costs, so the gains compound to ~3–4x on compute-bound layers with one joint fine-tune.
- **It is a hardware-conditional lever.** 2:4 pays on Ampere+ Sparse Tensor Cores (datacenter GPUs, Jetson Orin) and on compute-bound layers. On phone NPUs, Edge-TPUs, and microcontrollers there is no Sparse Tensor Core and 2:4 buys you nothing but a smaller file — quantize instead.
- **Always measure on your target.** Warm up, synchronize, lock clocks, and check the per-layer kernel choice; a silent dense fallback (wrong shape, wrong dtype, wrong operand orientation) is the most common reason a 2:4 model shows no speedup.

## Further reading

- **Mishra, Latorre, Pool, Stosic, Stosic, Venkatesh, Yu, Micikevicius — "Accelerating Sparse Deep Neural Networks" (NVIDIA, 2021).** The foundational 2:4 paper: the pattern, the compressed format, the prune-and-retrain recipe, and the accuracy-recovery results across vision and language models.
- **Pool and Yu — "Channel Permutations for N:M Sparsity" (NeurIPS 2021).** How permuting channels before masking lowers 2:4 pruning error for free at inference.
- **Frantar and Alistarh — "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot" (2023).** One-shot second-order pruning to 2:4 for LLMs without retraining; the practical route at billion-parameter scale.
- **Sun, Liu, Bair, Kolter — "A Simple and Effective Pruning Approach for Large Language Models" (Wanda, 2023).** A cheap activation-aware alternative for 2:4 on LLMs.
- **NVIDIA ASP / Apex `apex.contrib.sparsity` docs** and the **PyTorch semi-structured sparsity** guide (`torch.sparse.to_sparse_semi_structured`) — the two code paths in this post.
- **NVIDIA cuSPARSELt and TensorRT documentation** — the production sparse-GEMM library and the `--sparsity=enable` build path.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up) for the coarse-but-universal alternative, [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) to know if you are compute- or memory-bound, [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) to know whether your target even has a Sparse Tensor Core, and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for assembling all the levers for a real deployment.
