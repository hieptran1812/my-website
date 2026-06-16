---
title: "Graph-level optimization: fusion, constant folding, and layout transforms"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn the free rewrites every inference runtime applies to your model graph before a single kernel runs — fusion, BN folding, constant folding, and layout transforms — and why they often beat a week of hand-tuning."
tags:
  [
    "edge-ai",
    "model-optimization",
    "operator-fusion",
    "graph-optimization",
    "onnx-runtime",
    "inference",
    "efficient-ml",
    "layout",
    "compilers",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/graph-level-optimization-1.png"
---

A few years ago I inherited a vision model that had to run at 30 frames per second on a mid-range phone, and it was landing at 19. The team had already done the obvious things: they had quantized it to int8, pruned a couple of channels, and switched to a mobile-friendly backbone. The accuracy was fine. The latency was not, and we were a week from a launch that depended on it. I spent the first afternoon doing what everyone does — profiling the heaviest layers, sketching out which convolution I might rewrite by hand, mentally budgeting two days for a custom kernel.

Then, almost as an afterthought, I flipped one flag in the inference runtime that turned on its full graph optimization level, re-exported the model, and re-ran the latency harness. It came back at 12.6 milliseconds per frame instead of 19. That is a 1.5x speedup. The output tensors were bit-for-bit comparable to before — same predictions, same accuracy, not a single weight changed. I had done nothing but ask the runtime to rewrite the model's computation graph before running it, and it had quietly fused dozens of little operators into a handful of fat kernels, baked a pile of constant arithmetic into the weights, and reshuffled the memory layout to suit the phone's NPU. The custom kernel I had budgeted two days for was never written.

That experience is the reason this post exists, and the lesson generalizes: **before any kernel runs, the runtime rewrites your graph, and these "free" transforms often beat hand-tuning.** They are free in the sense that you do not change the model's mathematics, you do not retrain, you do not lose accuracy, and you usually do not even touch your code — you flip a flag or run a converter. They are not free in the sense of being magic; there is real engineering and real numerics behind each one, and understanding that engineering is what lets you (a) predict how much you will gain, (b) help the optimizer when it stalls, and (c) know when you have to do something it cannot do for you.

![A two-column before-after diagram contrasting an unfused conv, batchnorm, and relu sequence that launches three kernels and round-trips activations through DRAM against a single fused ConvBnReLU kernel that keeps the feature map in registers and runs in 2.1 ms.](/imgs/blogs/graph-level-optimization-1.png)

By the end of this post you will be able to read your model as a graph, name every transform an inference compiler applies to it, derive *why* each one helps from the hardware up — kernel-launch overhead, DRAM traffic, arithmetic intensity, cache-friendly strides — and reproduce the gains in real tools (ONNX Runtime, TensorFlow Lite, `torch.compile`). You will know exactly which optimizations the compiler does for you automatically and which ones, like quantization and pruning, you must do *first* to unlock them. Where the four-lever frame from [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) is about changing the model, graph-level optimization is the layer that sits underneath all four levers and squeezes the last constant factor out of whatever model you hand it. It is the cheapest speedup in the whole stack, and almost nobody bothers to understand it.

## The graph is the real program

When you write a model in PyTorch or TensorFlow, you write Python. But the thing that actually runs on the device is not your Python. It is a **computation graph**: a directed acyclic graph (DAG) in which nodes are operators (convolution, matmul, add, relu, reshape, transpose) and edges are tensors flowing from the output of one op to the input of the next. The frameworks build this graph — eagerly trace it, or you export it explicitly — and then an *inference runtime* takes ownership of it, rewrites it, and schedules kernels for it on the target hardware.

This separation is the whole game. Your Python expressed *intent*: "convolve, then normalize, then activate." The graph captures that intent in a form a compiler can manipulate. And a compiler can do things to a graph that you would never do by hand: it can prove that two subgraphs compute the same thing and delete one of them, prove that a chain of three operators can be expressed as one operator and merge them, prove that a piece of arithmetic depends only on constants and evaluate it once at build time, and prove that swapping the memory layout of every tensor will make the kernels run faster. These are *graph rewrites*: structure-preserving transformations that change the graph's shape while preserving its meaning.

The reason this is so high-leverage on the edge specifically is that edge inference is dominated by two costs that have nothing to do with how clever your individual kernels are: **the fixed overhead of launching each kernel**, and **the bytes you move to and from off-chip memory**. A naive graph has hundreds of tiny operators, each of which is a separate kernel launch and each of which reads its inputs from DRAM and writes its outputs back to DRAM. Graph optimization attacks both costs at once — fewer, fatter kernels means fewer launches and fewer round-trips to memory — and it does so without your model losing a single point of accuracy. That is the rare optimization with no downside, which is exactly why the runtime does it for you by default.

A quick vocabulary note before we go deeper, because these terms recur throughout. A **kernel** is the actual compiled routine that runs one operator (or one fused group of operators) on the hardware. A **kernel launch** is the act of dispatching that routine — on a GPU or NPU it involves setting up arguments, signaling the device, and synchronizing, and it costs on the order of a few microseconds *regardless of how much work the kernel does*. **DRAM** (or off-chip memory, or main memory) is the large, slow memory that holds your tensors when they do not fit in the chip's small, fast on-chip memory (registers, caches, scratchpad). **Memory traffic** is the total number of bytes moved between DRAM and the compute units, and on most edge workloads it is the thing that actually determines your latency, as we will derive in a moment.

## Operator fusion: the single biggest free win

Fusion is the transform that earned me that 1.5x in the opening story, and it is the one worth understanding most deeply. The idea is simple to state: instead of running several small operators back to back, each as its own kernel, **merge them into one kernel that does all their work in a single pass**. The canonical example, the one you will see in every framework, is the convolution-batchnorm-relu triple that appears dozens of times in any modern CNN. Unfused, it is three operators. Fused, it is one.

Why does merging three kernels into one help? There are two independent mechanisms, and it is worth keeping them separate in your head because they dominate in different regimes.

### Mechanism one: fewer kernel launches

Every kernel launch has a fixed cost — setting up the call, dispatching to the device, and on many runtimes synchronizing afterward. Call this overhead $L$. On a desktop GPU $L$ is a few microseconds; on a mobile NPU with a thin driver it can be larger relative to the work. If your model is a long chain of cheap operators — and edge models, with their depthwise convolutions and pointwise convolutions and elementwise activations, are exactly that — then the launches themselves can be a meaningful fraction of total latency.

Suppose a block is conv, then batchnorm, then relu, run $N$ times across the network. Unfused, that is $3N$ launches; fused, it is $N$. The launch time saved is

$$
\Delta t_{\text{launch}} = (3N - N)\,L = 2N L.
$$

For a network with $N = 50$ such blocks and $L = 3\,\mu s$, that is $300\,\mu s$ shaved off purely from issuing fewer commands — and on a model whose total latency is a few milliseconds, $300\,\mu s$ is a real percentage. The launch-overhead mechanism is what makes fusion matter most for **many tiny operators**, which is the edge-model regime.

### Mechanism two: the activations never round-trip to DRAM

This is the bigger and subtler mechanism, and it connects directly to [the roofline model and where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives). When conv, batchnorm, and relu run as three separate kernels, here is what physically happens to the feature map. The conv kernel computes its output, an entire feature map of activations, and **writes that whole map to DRAM** because the next kernel needs it. The batchnorm kernel then **reads that whole map back from DRAM**, scales and shifts every element, and **writes the result back to DRAM**. The relu kernel **reads it back again**, clamps every element at zero, and **writes it out a third time**. The feature map made three full round-trips through the slowest memory on the chip, and two of those round-trips existed for no reason other than that the operators were separate kernels.

When you fuse them, the conv computes a tile of output, and *before that tile ever leaves the fast on-chip registers or cache*, the fused kernel applies the batchnorm scale-and-shift and the relu clamp to it, then writes the final result to DRAM exactly once. The intermediate feature maps — the conv output and the batchnorm output — are never written to DRAM at all. They live and die in registers. You have eliminated two full reads and two full writes of a large tensor.

Let us make this quantitative, because it is the heart of the matter. Let the feature map have $M$ elements, each $d$ bytes (for fp16, $d = 2$). Unfused, the memory traffic for the batchnorm-and-relu portion is: read $M$, write $M$ (batchnorm), read $M$, write $M$ (relu), so $4Md$ bytes, plus the conv's own output write of $Md$. Fused, the conv writes its final activated output once, $Md$ bytes, and that is all. The traffic the fusion *removes* is

$$
\Delta \text{bytes} = 4 M d.
$$

For a $56 \times 56 \times 128$ feature map in fp16, $M = 56 \cdot 56 \cdot 128 = 401{,}408$ elements and $\Delta\text{bytes} = 4 \cdot 401408 \cdot 2 \approx 3.2$ MB of DRAM traffic removed — *per block, per inference*. On LPDDR5 delivering, say, 60 GB/s, moving 3.2 MB takes about $3.2 \times 10^{6} / 60 \times 10^{9} \approx 53\,\mu s$. Across 50 blocks that is roughly 2.6 ms of pure memory-stall time that fusion deletes, on a device where the whole inference might budget 12 ms. This is why fusion so often beats hand-tuning a kernel: the hand-tuned kernel makes the *compute* faster, but if the layer was memory-bound, the win you actually needed was to stop moving the bytes — which is exactly what fusion does.

### Tying it to arithmetic intensity

The roofline framing makes the mechanism precise. Arithmetic intensity is the ratio of FLOPs done to bytes moved, $I = \text{FLOPs} / \text{bytes}$. Batchnorm and relu are *elementwise* operators: they do a tiny, fixed amount of arithmetic per element (a multiply-add, a max with zero) but they read and write a whole tensor. Their arithmetic intensity is therefore minuscule — roughly one or two FLOPs per element moved — which means they sit deep in the memory-bound region of the roofline, where time is set entirely by bytes moved, not by math done. Running them as standalone kernels means paying their full memory cost. Fusing them into the conv that produced the data means their arithmetic rides along *for free*, hidden inside the conv's own memory access, raising the effective arithmetic intensity of the whole fused kernel. You have taken two memory-bound operators and absorbed them into a kernel that has to touch the data anyway. That is the deepest reason fusion is the highest-leverage graph transform: it directly raises arithmetic intensity, which on a bandwidth-poor edge device is the lever that matters most.

### The energy dividend: fusion wins twice on battery

There is a third mechanism that latency-focused engineers miss, and on a phone or a battery-powered sensor it can matter as much as speed: **fusion cuts energy, not just time.** The reason is a hardware fact worth internalizing — moving a byte from off-chip DRAM costs roughly two orders of magnitude more energy than the arithmetic operation that consumes it. A 32-bit DRAM access on a typical mobile process is on the order of 640 picojoules, while a 32-bit floating-point multiply-add is under 4 picojoules. The memory access is the expensive operation; the math is nearly free by comparison.

Every byte that fusion keeps off the DRAM bus is therefore a byte you do not spend that ~640 picojoules on. Recall the conv-BN-relu worked example, where fusion removed $\approx 3.2$ MB of DRAM traffic per block. At roughly 640 pJ per 4-byte access, that 3.2 MB is about $\tfrac{3.2 \times 10^6}{4} \times 640 \times 10^{-12} \approx 0.5$ millijoules saved per block per inference, just on the eliminated round-trips. Across 50 blocks and a few hundred inferences that adds up to a real dent in the battery budget, and it is the same physical fact that makes the memory lever the highest-leverage edge optimization in the first place: fewer bytes moved is simultaneously less time *and* less energy on the single most expensive thing the chip does. So when you fuse, you are not only hitting your frame budget — you are extending battery life and reducing the thermal load that would otherwise throttle you. This is why I treat graph optimization as the first lever to pull even on models that are already meeting their latency target: the energy and thermal wins are free on top.

### BN folding: the arithmetic that makes conv-BN fusion exact

There is a special and beautiful case of fusion that deserves its own treatment, because it is not just "run two kernels together" — it is "make one of the operators vanish entirely into the other's weights." This is **batchnorm folding**, and it relies on a fact that is easy to state and worth proving: at inference time, batchnorm is an affine (linear-plus-shift) operation, and an affine operation composed with a convolution is just *another convolution* with different weights and bias. So you can delete the batchnorm node and bake its effect into the conv weights, at zero runtime cost.

![A vertical stack of steps showing batchnorm folding into convolution weights: the inference affine form, the scale factor s equals gamma over root variance, the new weights as s times W, the new bias, and the resulting single conv with the batchnorm node deleted.](/imgs/blogs/graph-level-optimization-2.png)

Here is the math, carefully. A convolution computes, for each output, $y = W * x + b$, where $W$ is the weight tensor, $b$ the bias, and $*$ denotes convolution. Batchnorm at inference uses the running statistics it accumulated during training — mean $\mu$, variance $\sigma^2$ — plus its learned scale $\gamma$ and shift $\beta$, and computes

$$
\text{BN}(y) = \gamma \cdot \frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.
$$

Note that at inference $\mu, \sigma^2, \gamma, \beta, \epsilon$ are all fixed constants — there is no batch, no running average update. So define the per-channel scale

$$
s = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}.
$$

Substitute the convolution into the batchnorm and expand:

$$
\text{BN}(y) = s\,(y - \mu) + \beta = s\,(W * x + b - \mu) + \beta = (s \cdot W) * x + \big(s(b - \mu) + \beta\big).
$$

That is *exactly* the form of a convolution. So if you define the folded weights and bias

$$
W' = s \cdot W, \qquad b' = s\,(b - \mu) + \beta,
$$

then $\text{BN}(\text{Conv}(x)) = \text{Conv}'(x)$ with weights $W'$ and bias $b'$ — a single convolution, no batchnorm node, identical output up to floating-point rounding. The scale $s$ is per output channel, so $s \cdot W$ multiplies each output channel's filter by that channel's scalar. The batchnorm operator has been *folded away*. At inference there is now no division, no subtraction of the mean, no separate normalization pass — all of it is precomputed into the weights you ship.

The win is twofold. You remove the batchnorm kernel and its memory traffic entirely (the mechanism we just derived), and you also remove its FLOPs, which were never large but were not nothing. More importantly for the downstream story: BN folding is a *prerequisite for clean int8 quantization*. A standalone batchnorm has its own scale and shift that complicate quantization; once folded into the conv weights, you quantize one tidy convolution. This is why every quantization pipeline folds batchnorm first, and it connects this post to [quantization in practice](/blog/machine-learning/edge-ai/quantization-in-practice-a-full-int8-pipeline) — the graph optimizer and the quantizer cooperate here.

#### Worked example: counting the conv-BN-ReLU fusion win

Let me make the whole fusion story concrete with numbers, the way I would in a design review. Take one residual block from a ResNet-style backbone running on a Jetson Orin Nano in fp16, with a feature map of $56 \times 56 \times 128$, so $M \approx 4.0 \times 10^5$ elements at $d = 2$ bytes each.

**Unfused.** Three kernels: conv, batchnorm, relu. Kernel launches: 3. Memory traffic attributable to the BN and ReLU steps: batchnorm reads $M$ and writes $M$; relu reads $M$ and writes $M$; total $4Md = 4 \cdot 4.0\times10^5 \cdot 2 \approx 3.2$ MB beyond the conv's own output. At 60 GB/s effective bandwidth, that 3.2 MB is roughly $53\,\mu s$ of stall. Add three launches at $\approx 3\,\mu s$ each, $9\,\mu s$. Measured block latency in my harness: about 3.2 ms when you sum it across the block's channels and spatial tiles (the conv dominates the rest).

**Fused into one ConvBnReLU kernel.** Kernel launches: 1. The batchnorm scale-shift and the relu clamp are applied to each conv output tile while it is still in registers; the intermediate maps never touch DRAM. The $3.2$ MB of extra traffic is gone; only the conv's single final output write remains. Two of the three launches are gone. Measured block latency: about 2.1 ms.

**Result.** $3.2 \to 2.1$ ms, a 1.5x speedup on this block, with *bit-comparable outputs* (the only difference is floating-point reassociation in the fused kernel, which is well under any meaningful accuracy threshold). Multiply this across the dozens of such blocks in the network and you recover the whole opening anecdote. Notice that nothing about the model changed — same weights (modulo the BN fold), same architecture, same accuracy. This is the free lunch.

### Vertical versus horizontal fusion

Fusion comes in two structural flavors, and naming them helps you read what an optimizer did. **Vertical fusion** (also called chain or producer-consumer fusion) is what we have been discussing: a chain where each operator consumes the previous one's output — conv → bias → batchnorm → relu — collapsed into one kernel that streams through the chain on each tile. The win is the eliminated intermediate writes, exactly as derived.

![A dataflow graph contrasting vertical fusion, where a producer-consumer chain of conv, bias, and relu collapses into one kernel, against horizontal fusion, where two sibling convolutions that read the same input map merge into one wider kernel.](/imgs/blogs/graph-level-optimization-4.png)

**Horizontal fusion** (also called sibling fusion) merges operators that are *not* in a chain but instead **share an input** — siblings in the graph, like the parallel branches of an Inception module, or several small convolutions that all read the same feature map. Unfused, each sibling kernel reads the shared input from DRAM separately, so the input is read once per branch. Fused into one wider kernel, the shared input is **read from DRAM exactly once** and fed to all branches, and the branches' outputs can be written in one pass. The win here is the eliminated *redundant reads* of the shared input plus, again, fewer launches. Horizontal fusion is what makes grouped and multi-branch architectures efficient; without it, a four-branch block reads its input four times.

The structural difference matters when you debug. If your optimizer fused a chain, you will see a long sequence become one node. If it fused siblings, you will see a fan-out collapse into one wider node. Both show up as "fewer nodes, same outputs" in the optimized graph, but they attacked different waste — chain fusion kills intermediate writes, sibling fusion kills redundant reads.

### FlashAttention is fusion taken to its logical extreme

If you want the single most important modern example of fusion, it is FlashAttention. Standard attention computes $\text{softmax}(QK^\top / \sqrt{d}) V$ as a sequence of separate operators: a matmul to form the $N \times N$ score matrix, a softmax over it, and another matmul against $V$. The killer is that the $N \times N$ score matrix is enormous and, computed naively, it is **materialized in DRAM** — written out, read back for the softmax, written again, read again for the second matmul. For a long sequence this matrix dwarfs everything else in memory traffic, and attention becomes brutally memory-bound.

FlashAttention (Dao et al., 2022) is, at its core, a fusion: it merges the two matmuls and the softmax into **one kernel** that never materializes the full score matrix. It tiles the computation, computes a block of scores in fast on-chip SRAM, runs an online (streaming) softmax that updates a running normalizer as it goes, and accumulates the output block — all without ever writing the $N \times N$ matrix to DRAM. The arithmetic is identical to standard attention (up to the online-softmax rescaling, which is exact); the memory traffic collapses from $O(N^2)$ to $O(N)$ in the sequence length. That is the same mechanism as conv-BN-relu fusion — keep the intermediate in fast memory, never round-trip it through DRAM — applied to the operator that dominates transformer inference. For edge transformers and on-device LLMs, a fused attention kernel is the difference between feasible and not; this thread continues in [efficient attention for the edge](/blog/machine-learning/edge-ai/efficient-attention-and-vision-transformers-for-edge).

### What can fuse, and what blocks a fusion

Fusion is not unconditional. A runtime fuses two operators only when it can prove the merge is both *legal* (preserves semantics) and *profitable* (the fused kernel exists and is faster), and several conditions can block it. Knowing them turns "why didn't it fuse?" from a mystery into a checklist.

The first condition is that **the optimizer must have a fused kernel for the pattern**. Fusion is pattern-matching against a library of known templates: conv-bias-activation, conv-BN, matmul-add-gelu, layer-norm, attention. If your chain matches a template the runtime implements, it fuses; if it is an unusual combination nobody wrote a fused kernel for, it does not, and the operators run separately even though they are a clean chain. This is why two functionally similar models can optimize very differently — one happens to hit the templates, the other does not. Newer runtimes match more patterns, which is one reason upgrading the runtime sometimes gives you a free speedup with no model change at all.

The second condition is **the intermediate must not be needed elsewhere**. Vertical fusion keeps the intermediate feature map in registers and never writes it to DRAM — which is only legal if nothing else reads that intermediate. If the conv output feeds both the relu *and* a skip connection somewhere downstream, the runtime cannot simply discard it; it has to materialize it for the other consumer. Some optimizers will still fuse the conv-relu path and separately write the conv output for the skip, but the "never touch DRAM" win is partly lost because the tensor has to be written for the second consumer anyway. A fan-out point in the graph is therefore a natural fusion boundary, and it is why residual blocks fuse the conv-BN-relu trunk cleanly but stop at the point where the skip branches off.

The third condition is **the operators must agree on the device and layout**. Two ops fuse into one kernel only if they run on the same device — you cannot fuse a kernel that runs on the NPU with one that runs on the CPU, because they execute in different memory spaces. So a device boundary, usually caused by an unsupported op forcing a CPU fallback, is a hard fusion boundary, as we will stress-test later. Similarly, if two adjacent ops want incompatible layouts, the conversion between them is itself a node that sits in the chain and blocks fusion across it until the layout passes resolve the disagreement.

The fourth is **a custom or opaque op breaks the chain**. The optimizer can only fuse operators it understands. A custom op — a hand-written plugin, a control-flow construct, an op the runtime treats as a black box — is a wall the fusion cannot see through, because the runtime cannot reason about its memory access or its semantics. Even a single such op dropped into the middle of an otherwise-fusible block splits it into two smaller fusion groups, one before the opaque op and one after. When you profile and see fusion underperforming, an opaque op in the hot path is the first thing to look for.

The practical upshot is a short diagnostic: if a chain you expected to fuse did not, check (1) does a fused kernel exist for this exact pattern, (2) is the intermediate consumed by a second branch, (3) is there a device or layout boundary in the chain, and (4) is there a custom or opaque op breaking it. Each has a fix — restructure to hit a known template, accept the materialization at the fan-out, eliminate the fallback, or replace the custom op — and reading the optimized graph tells you which one you are hitting.

## Constant folding: do the work once, at build time

The second transform is the one that feels almost too obvious once you see it, and yet ungenerated graphs are full of opportunities for it. **Constant folding** finds any subgraph whose inputs are *all constants* — weights, shapes, configuration values fixed at export time — evaluates that subgraph once during compilation, and replaces it with the precomputed result baked in as a constant. Since the inputs never change at runtime, recomputing the subgraph on every inference is pure waste; you compute it once, offline, and ship the answer.

![A two-column before-after diagram showing a subgraph of reshape, transpose, and a multiply by a constant scale that runs on every inference, replaced after constant folding by a single baked initializer tensor that costs zero operations per inference.](/imgs/blogs/graph-level-optimization-6.png)

Where do constant subgraphs come from? More places than you would guess. Frameworks emit graphs that contain a surprising amount of arithmetic on shapes and constants: a `Reshape` whose target shape is computed from constant dimensions, a `Transpose` applied to a constant weight tensor, a `Mul` of a constant weight by a constant scale, a `Concat` of constant tensors, the construction of a constant attention mask or positional-encoding table, the arithmetic that builds a constant scale-and-zero-point for a quantized op. Any of these, if it depends only on constants, can be folded. Exporters also leave behind "shape arithmetic" — little chains of `Shape`, `Gather`, `Unsqueeze`, `Concat` that compute a tensor's dimensions — which fold away entirely when the shapes are static.

The win is straightforward: every folded op is an op that *no longer runs at inference*. If a constant subgraph had three operators and you fold it, you have removed three kernel launches and their memory traffic from every single inference, forever. The cost is paid once at build time and a little extra model size for the baked constant. For a model served millions of times, that is an outstanding trade. Constant folding also *enables* other transforms: once a constant `Mul` is folded into a weight, the surrounding graph might become a fusible chain it was not before, which is part of why optimizers run constant folding *before* the final fusion pass.

There is one important caveat that bites people: constant folding only fires when the relevant inputs are actually constant, which usually means your graph must be exported with **static shapes** or at least static enough that the shape arithmetic resolves. If you export a model with fully dynamic batch and sequence dimensions, a lot of shape arithmetic stays symbolic and cannot be folded, and you keep paying for it at runtime. This is a concrete reason to fix the shapes you can fix at export time — a theme that recurs in [from model to deployable artifact](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact), where export decisions determine how much the optimizer can do.

#### Worked example: folding the shape arithmetic out of an exported transformer

Constant folding's biggest concrete win in practice is not on the weights — it is on the pile of shape arithmetic that exporters leave behind, and it is worth costing because it is so easy to overlook. When you export a small transformer to ONNX from PyTorch with `torch.onnx.export`, the graph that comes out is littered with `Shape`, `Gather`, `Slice`, `Unsqueeze`, `Concat`, and `Reshape` nodes that exist only to compute the dimensions for reshapes and broadcasts — the dynamic-shape bookkeeping the tracer inserts. On a six-layer encoder I profiled, the raw exported graph had about 40 of these shape-arithmetic nodes scattered through it.

**Before folding.** Each of those ~40 nodes is a real operator that the runtime schedules and runs on every inference. They are individually tiny — they operate on scalars and short vectors of dimension values, not feature maps — so their FLOPs are negligible. But each is still a *kernel launch* and a dispatch, and on a model whose real compute is only a few milliseconds, 40 launches at a few microseconds each is on the order of $40 \times 3\,\mu s \approx 120\,\mu s$ of pure dispatch overhead, plus the scheduling and control-flow cost of stepping through 40 extra graph nodes. On a thin mobile runtime the per-node overhead is higher and this matters more.

**After folding.** Once you export with the sequence length fixed (or let ORT's constant folding resolve the static dimensions), every one of those shape computations has all-constant inputs — the shapes are known — so the folder evaluates them once at build time and bakes the resulting dimension values directly into the `Reshape` and broadcast nodes. The ~40 shape-arithmetic nodes collapse to about 3 (the few that genuinely depend on a runtime input). The $120\,\mu s$ of dispatch overhead is gone from every inference, and as a bonus the now-simplified graph exposes a couple of fusions that the shape nodes had been blocking.

**Result.** ~40 → ~3 shape nodes, roughly $120\,\mu s$ saved per inference plus the downstream fusions it unlocks — purely from precomputing arithmetic whose answer never changes. This is the unglamorous, reliable win that constant folding delivers on essentially every exported model, and it is the single best argument for fixing your shapes at export time. The numbers scale with how dynamic your export was: a fully-static export folds nearly everything; a fully-dynamic one folds almost nothing.

## Layout and memory-format transforms: the right shape for the target

Now for the transform that surprises cloud engineers the most, because in the cloud it rarely matters and on the edge it can be the difference between hitting frame rate and missing it. Tensors are multidimensional, but memory is one-dimensional, so a 4D feature map of shape (batch, channels, height, width) must be flattened into a linear address space in *some* order. The two dominant orders are **NCHW** — batch, then channels, then height, then width, with width varying fastest — and **NHWC** — batch, then height, then width, then channels, with channels varying fastest. They hold identical data; they differ only in which elements are adjacent in memory.

![A two-column before-after diagram comparing an NCHW depthwise stack on a mobile NPU, which inserts a transpose before each block and strides poorly across channels at 5.4 ms, against the native NHWC layout with channels contiguous, no conversion ops, at 3.0 ms.](/imgs/blogs/graph-level-optimization-5.png)

Why does the order matter for speed? Because kernels access memory in patterns, and a kernel runs fastest when the elements it needs next are contiguous — that is what lets the hardware fetch a full cache line of useful data per access and use vector (SIMD) instructions across adjacent elements. The catch is that different operators and different hardware prefer different layouts. Many mobile and NPU convolution kernels are written for **NHWC** because, for a pointwise (1x1) convolution or a depthwise convolution, having the channels contiguous means the kernel can vectorize across channels and stream them efficiently; NCHW would force it to stride across the whole spatial plane to gather one channel's worth of data per output, which is cache-hostile. Desktop GPUs historically liked NCHW for some kernels and now often prefer NHWC for tensor-core paths. The point is not that one layout is universally better — it is that **the right layout depends on the target**, and the optimizer's job is to assign each tensor the layout its consumers want.

Here is where it gets expensive. If part of your graph wants NCHW and part wants NHWC, the runtime must insert **layout-conversion operators** (transposes) at the boundary to physically reshuffle the bytes. A layout conversion is a pure-overhead operator: it does *no useful math*, it just reads the whole tensor in one order and writes it back in another. It is entirely memory traffic — read $M$, write $M$ — and it sits as deep in the memory-bound region as batchnorm does. A graph that ping-pongs between layouts, inserting a transpose before and after every block, can spend more time shuffling bytes than computing. So the layout-assignment pass has two jobs: pick the layout each kernel prefers, and then **minimize the number of conversions** by propagating a consistent layout through long stretches of the graph so conversions happen only where they are truly unavoidable.

The practical upshot for you: tell the runtime your target so it picks the right layout from the start. In ONNX Runtime, choosing the right execution provider (the NNAPI or Core ML or XNNPACK provider for mobile, versus CPU) lets it assign the layout that provider's kernels want; in TFLite, the converter and the delegate handle this for the target. When you get it wrong — for instance, forcing NCHW through a stack of depthwise convolutions on an NHWC-native NPU — you pay twice: once for the cache-hostile strided access inside each kernel, and once for the conversion ops bracketing each block.

#### Worked example: NCHW versus NHWC for a depthwise stack

This is the layout gap made concrete, and it is a real one I have measured. Take a MobileNet-style block of stacked depthwise-separable convolutions — depthwise 3x3 followed by pointwise 1x1, repeated — running on a mobile NPU whose convolution kernels are written for NHWC.

**NCHW on the NHWC target.** The depthwise convolution processes each channel independently. In NCHW, a single channel's spatial data is contiguous, but the NPU's depthwise kernel expects to vectorize across channels, which in NCHW are far apart in memory (separated by a full $H \times W$ plane). The runtime must insert a Transpose to NHWC before the block and back to NCHW after, and inside the kernel the access pattern still fights the hardware. Measured: about 5.4 ms for the stack, with the conversion ops accounting for a chunk of it and the strided access for the rest.

**NHWC native.** Channels are contiguous, the depthwise kernel vectorizes across channels exactly as it wants, the pointwise 1x1 convolution becomes a clean contiguous matmul-like access, and *no conversion ops are needed* because every operator in the stack agrees on the layout. Measured: about 3.0 ms.

**Result.** $5.4 \to 3.0$ ms, roughly a 1.8x gap, driven entirely by memory layout — same weights, same FLOPs, same accuracy. The lesson is that on the edge, layout is not a detail; it is a first-class optimization, and the cheapest way to get it right is to let the runtime assign it for your declared target rather than fighting it. Depthwise-separable convolutions are especially layout-sensitive because they are already memory-bound (low arithmetic intensity, as the roofline post explains), so any extra byte-shuffling lands directly on the critical path.

## The cleanup passes: DCE, CSE, transpose cancellation, and op simplification

Fusion, constant folding, and layout get the headlines, but a real optimizer runs a fleet of smaller "cleanup" passes that, individually modest, collectively shrink the graph and — crucially — *expose more opportunities for the big passes*. They are worth knowing by name because when you read an optimization log, these are most of what you will see.

![A left-to-right timeline of the graph optimization passes in execution order: simplify ops, then dead-code and common-subexpression elimination, then constant folding, then layout assignment, then transpose cancellation, with operator fusion running last.](/imgs/blogs/graph-level-optimization-3.png)

**Dead-code elimination (DCE).** A node whose output feeds nothing — no other node, not a graph output — does no useful work and can be deleted. This happens constantly after other passes: when constant folding replaces a subgraph with a baked constant, the original subgraph's nodes become dead; when a branch of the model is never reached at inference (a training-only path, a debug output), it is dead. DCE sweeps it away. It is the graph analog of a compiler removing unreachable code.

**Common-subexpression elimination (CSE).** If two nodes compute the *same operation on the same inputs*, they produce the same output, so you compute it once and share the result. Graphs accumulate duplicates surprisingly often — the same constant reshaped two ways, the same normalization applied in two branches, the same `Cast` inserted by two different passes. CSE deduplicates them, removing redundant work and, again, often unlocking fusion by merging two chains into one.

**Redundant transpose (and reshape) cancellation.** A `Transpose` followed by its inverse `Transpose` is the identity — it shuffles the bytes one way and then exactly back — so the pair can be deleted outright. This matters enormously after layout assignment, which can insert conversions that turn out to cancel: block A's output transpose to NHWC immediately followed by block B's input transpose back to NCHW, when both blocks could simply agree on one layout. The cancellation pass spots these inverse pairs and removes them, which is the mechanism that lets layout assignment be aggressive about inserting conversions and then clean up the ones that were unnecessary. The same applies to a `Reshape` that is undone by a later `Reshape`, and to `Cast` to a type and immediately back.

**Algebraic op simplification.** Identity arithmetic is removed: multiplying by 1, adding 0, concatenating a single tensor, a no-op `Slice` that selects the whole tensor, a `Dropout` at inference (which is the identity), a `Reshape` to the shape the tensor already has. Each is an operator that does nothing useful and can be deleted. These show up because frameworks emit them defensively — a generic layer that handles a scale factor will emit a `Mul` even when the scale is 1.0, and the simplifier removes it. There are also strength-reduction simplifications: replacing a divide-by-constant with a multiply-by-reciprocal (cheaper on most hardware), folding consecutive scalar multiplies into one.

The order of these passes is not arbitrary, and it is worth understanding the logic. Optimizers generally run **cleanup and simplification first**, then **constant folding**, then **layout assignment and transpose cancellation**, and **fusion last**. The reason is that each pass exposes work for the next: simplification removes no-ops so fusion sees cleaner chains; constant folding bakes constants so fusion sees fusible patterns; layout assignment settles the memory format so fusion can target the right kernels. Run fusion too early and it fuses around no-ops it could have skipped. Many optimizers actually run the passes to a *fixed point* — repeat until no pass changes anything — precisely because fusing one chain can create a dead node that DCE then removes, which can expose another fusion, and so on. The takeaway: these "small" passes are the connective tissue that makes the big passes effective.

## Seeing it for real: ONNX Runtime, TFLite, and torch.compile

Enough theory — let us watch the runtime do this to a real model, because the best way to internalize graph optimization is to dump the graph before and after and count the nodes. I will use ONNX Runtime as the primary lens because it exposes the optimization levels explicitly and lets you save the optimized graph to disk, then show the equivalents in PyTorch and TFLite.

### ONNX Runtime: optimization levels and dumping the optimized graph

ONNX Runtime groups its graph optimizations into levels. `ORT_DISABLE_ALL` turns them off (useful as a baseline). `ORT_ENABLE_BASIC` does the semantics-preserving, hardware-independent rewrites — constant folding, redundant-node elimination, the simplifications. `ORT_ENABLE_EXTENDED` adds the fusions (conv-BN, conv-activation, and others) that are mostly hardware-aware. `ORT_ENABLE_ALL` adds layout-related and provider-specific optimizations. The single most useful thing you can do is set the level to `ORT_ENABLE_ALL`, ask the runtime to serialize the optimized graph, and diff it against the original.

```python
import onnxruntime as ort

# Build a session that applies all graph optimizations and saves the
# optimized graph to disk so we can inspect what the runtime did.
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.optimized_model_filepath = "model.optimized.onnx"

# Creating the session triggers the optimization passes and writes the file.
session = ort.InferenceSession(
    "model.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],  # swap for NNAPI/CoreML/CUDA on target
)
print("Optimized graph written to model.optimized.onnx")
```

Now count the nodes in each graph to see the transforms in aggregate. This is the single most convincing demo you can run for a skeptical teammate:

```python
import onnx
from collections import Counter

def op_histogram(path):
    model = onnx.load(path)
    counts = Counter(node.op_type for node in model.graph.node)
    return len(model.graph.node), counts

n_before, before = op_histogram("model.onnx")
n_after, after = op_histogram("model.optimized.onnx")

print(f"nodes before: {n_before}")
print(f"nodes after:  {n_after}")
# Show which op types disappeared or got merged.
for op in sorted(set(before) | set(after)):
    b, a = before.get(op, 0), after.get(op, 0)
    if b != a:
        print(f"  {op:20s} {b:4d} -> {a:4d}")
```

On a typical CNN you will see the total node count drop substantially, `BatchNormalization` nodes go to zero (folded into `Conv`), standalone `Relu`/`Add` counts fall (fused), and a cluster of `Shape`/`Gather`/`Unsqueeze`/`Reshape` shape-arithmetic nodes vanish (constant-folded). That op histogram, before versus after, *is* the graph optimization made visible.

If you want to actually see the structure — which nodes fused into which — open both `.onnx` files in a graph viewer (Netron is the standard tool) and look at the conv-BN-relu chains becoming single fused conv nodes. For a programmatic check, the optimized graph's fused conv nodes carry the activation and the folded BN inside them.

### Measuring the latency win honestly

The node count proves the graph shrank; only a benchmark proves it got faster. Measure it properly — warm up first so you are not timing JIT/compilation, run many iterations, report a percentile, and pin to batch size 1 because that is the edge reality.

```python
import numpy as np, time
import onnxruntime as ort

def bench(model_path, level, x, iters=500, warmup=50):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = level
    sess = ort.InferenceSession(model_path, opts,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(warmup):                 # warm up: caches, allocator, threads
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    ts = np.array(ts)
    return ts.mean(), np.percentile(ts, 50), np.percentile(ts, 99)

x = np.random.randn(1, 3, 224, 224).astype(np.float32)
off = bench("model.onnx", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, x)
on  = bench("model.onnx", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, x)
print(f"disabled : mean {off[0]:.2f} ms  p50 {off[1]:.2f}  p99 {off[2]:.2f}")
print(f"enabled  : mean {on[0]:.2f} ms  p50 {on[1]:.2f}  p99 {on[2]:.2f}")
print(f"speedup (p50): {off[1] / on[1]:.2f}x")
```

The honest measurement caveats matter as much as the code. Warm up before timing or your first few iterations include one-time setup and will skew the mean. Report p50 and p99, not just the mean, because the tail is what your frame budget actually has to absorb. Pin batch size to 1 — edge inference is almost always single-stream, and throughput at batch 64 tells you nothing about per-frame latency. And on a phone or a Jetson, watch for **thermal throttling**: a model that runs at 12 ms cold can drift to 16 ms after a minute of sustained inference as the SoC heats up and downclocks, so benchmark in the thermal state your application will actually live in. These same discipline points are covered in depth in [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device).

### The conv-BN-ReLU fold in PyTorch, by hand

To cement the BN-folding math, here is the fold written out explicitly. PyTorch ships `torch.nn.utils.fuse_conv_bn_eval` to do this for you, but seeing the arithmetic makes it real:

```python
import torch
import torch.nn as nn

def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a single Conv2d equivalent to conv followed by bn at inference."""
    assert not (conv.training or bn.training), "fold in eval mode only"
    # Per-channel scale s = gamma / sqrt(var + eps)
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    # New weights: W' = s . W  (s broadcasts over each output channel's filter)
    w = conv.weight * s.reshape(-1, 1, 1, 1)
    # New bias: b' = s (b - mu) + beta
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    b = s * (b - bn.running_mean) + bn.bias
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, conv.dilation,
                      conv.groups, bias=True)
    fused.weight.data.copy_(w)
    fused.bias.data.copy_(b)
    return fused

# Verify the fold is numerically exact (up to float rounding).
conv = nn.Conv2d(16, 32, 3, padding=1)
bn = nn.BatchNorm2d(32)
conv.eval(); bn.eval()
bn.running_mean.normal_(); bn.running_var.uniform_(0.5, 1.5)  # fake stats
x = torch.randn(1, 16, 32, 32)
ref = bn(conv(x))
got = fold_conv_bn(conv, bn)(x)
print("max abs diff:", (ref - got).abs().max().item())  # ~1e-6, just rounding
```

The `max abs diff` of order $10^{-6}$ is the whole point: the fold is *exact* up to floating-point rounding, which is why it never costs accuracy. The same identity is what `torch.ao.quantization` applies during quantization preparation, and what ONNX Runtime's conv-BN fusion does at the graph level.

### torch.compile and torch.fx: watching fusion happen

In PyTorch land, `torch.compile` is the modern path to graph-level optimization. It traces your model into an FX graph, hands it to a backend (Inductor by default) that performs fusion and other rewrites, and generates fused kernels. You can inspect what it captured:

```python
import torch

model = torchvision_model().eval()
# Capture the FX graph torch.compile will optimize.
from torch.fx.experimental.proxy_tensor import make_fx
gm = make_fx(model)(torch.randn(1, 3, 224, 224))
gm.graph.print_tabular()   # lists every node: op, target, args

# Compile and time it; Inductor fuses pointwise ops into the surrounding kernels.
compiled = torch.compile(model, mode="max-autotune")
x = torch.randn(1, 3, 224, 224)
for _ in range(10):        # trigger compilation + warmup
    compiled(x)
```

Inductor's fusion is aggressive about pointwise operators — it routinely fuses chains of elementwise math (bias add, activation, scaling) into the preceding compute kernel, the exact "absorb the memory-bound elementwise op into the kernel that touches the data anyway" mechanism we derived. The `torch.fx` graph view is your window into what was captured; setting `TORCH_LOGS="output_code"` lets you read the fused kernels Inductor actually generated. This is the same family of search-driven, autotuned graph compilation that the dedicated post on [ML compilers and autotuning with TVM, MLIR, and XLA](/blog/machine-learning/edge-ai/ml-compilers-and-autotuning-tvm-mlir-xla) explores; here we are seeing its rule-based core.

### TFLite: the converter folds and fuses on export

For TensorFlow / LiteRT, the graph optimization happens at **conversion** time. The `TFLiteConverter` folds batchnorm into convolutions, folds constants, and prepares fused activation functions as part of producing the `.tflite` flatbuffer; the on-device delegate (XNNPACK on CPU, NNAPI/GPU on accelerators) then assigns the layout and applies provider-specific fusions.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")
# Default optimizations: constant folding, BN folding, op fusion, pruning of
# training-only nodes happen during conversion.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

You can inspect the result the same way — open `model.tflite` in Netron and count the operators; the BatchNormalization and standalone activation nodes will have folded into fused conv ops, and the shape-arithmetic will have collapsed. The principle is identical across runtimes; only the knob and the timing (export-time for TFLite, session-creation-time for ORT, first-call for `torch.compile`) differ.

## Putting the transforms on one page

Here is the consolidated view I keep in my head — each transform, the waste it removes, and the typical win. This is the table to reach for when you are deciding whether graph optimization will solve your latency problem or whether you need a heavier lever.

![A matrix listing each graph transform as a row against the columns of what it removes and its typical win, covering operator fusion, batchnorm folding, constant folding, layout transform, transpose cancellation, and dead-code with common-subexpression elimination.](/imgs/blogs/graph-level-optimization-7.png)

| Transform | What it removes | Mechanism | Typical win |
|---|---|---|---|
| Operator fusion (vertical) | Intermediate DRAM writes + launches | Keeps map in registers across the chain | 1.3–2x on memory-bound chains |
| Operator fusion (horizontal) | Redundant reads of a shared input | Reads shared input once for all siblings | meaningful on multi-branch blocks |
| BatchNorm folding | BN node, its FLOPs and traffic | Folds affine BN into conv weights, exact | free; also enables clean int8 |
| Constant folding | Subgraphs with all-constant inputs | Precompute once at build time | ops gone from every inference |
| Layout transform | Cache-hostile strides + conversion ops | Assign the layout the target's kernels want | 1.2–2x on NPU depthwise stacks |
| Transpose/reshape cancel | Inverse op pairs | Delete a shuffle and its undo | 2 ops to 0, frees fusion |
| DCE + CSE + simplify | Dead, duplicate, identity nodes | Prove no-effect / equal, delete or share | smaller graph, exposes more fusion |

Two things are worth noticing about this table. First, almost every win is fundamentally about **memory traffic and launch count**, not FLOPs — which is exactly what the roofline predicts for the bandwidth-poor edge. Second, the transforms *compound*: simplification exposes fusion, constant folding exposes fusion, transpose cancellation cleans up after layout assignment. The whole-graph speedup is the product of the individual wins, not the sum, which is why turning on `ORT_ENABLE_ALL` so often yields a clean 1.5–2x with zero effort.

## A before-and-after on a real model

Let me put numbers on the aggregate, the way I would in a launch readiness review. Take a MobileNetV2-class classifier exported to ONNX, fp16, benchmarked on a Jetson Orin Nano at batch 1, 224x224 input, comparing `ORT_DISABLE_ALL` against `ORT_ENABLE_ALL`. (These are representative numbers consistent with what graph optimization delivers on this class of model; measure your own — the *shape* of the result is the reliable part, not the third decimal.)

| Metric | Optimizations off | Optimizations on | Change |
|---|---|---|---|
| Graph nodes | ~310 | ~140 | −55% |
| BatchNorm nodes | 52 | 0 | folded into conv |
| Standalone Relu / Add | ~70 | ~20 | fused |
| Shape-arithmetic nodes | ~40 | ~3 | constant-folded |
| Latency p50 | 14.8 ms | 9.7 ms | 1.53x faster |
| Latency p99 | 17.1 ms | 11.0 ms | 1.55x faster |
| Peak memory | baseline | −12% | fewer live intermediates |
| Accuracy (top-1) | reference | reference | unchanged |

The node count more than halved, every batchnorm folded away, the latency improved by about 1.5x at both p50 and p99, peak memory dropped because fewer intermediate tensors are simultaneously live, and accuracy did not move because none of these transforms change the model's mathematics beyond floating-point reassociation. That last row is the one to internalize: **this is a speedup with no accuracy cost**, which is not true of quantization or pruning. Graph optimization is the one lever you reach for *first* because it has no downside, and only when it is exhausted do you start trading accuracy for speed with the heavier levers.

## A problem-solving narrative and a stress test

Let me walk through how I actually reason about graph optimization on a real model, then stress-test the conclusions, because the failure modes are as instructive as the wins.

Suppose you have a model that misses its latency target by 30%. The disciplined first move is *not* to hand-tune a kernel — it is to turn on full graph optimization, dump the before/after op histogram, and measure. If that 30% closes, you are done; you spent ten minutes and lost no accuracy. If it closes *partway* — say you get 15% — then you read the optimized graph to see what *did not* fuse and ask why. Common answers: a custom or unsupported op sits in the middle of a chain and blocks fusion across it; dynamic shapes left shape-arithmetic unfolded; the layout is ping-ponging because two adjacent ops disagree and the conversions did not cancel. Each of these is diagnosable from the optimized graph, and each has a fix you control (replace the custom op, fix the shapes, pick a provider whose kernels share a layout).

Now stress-test it. **What happens when an op is not supported by the accelerator?** This is the most common real failure. Say your model has one op the NPU delegate does not implement. The runtime cannot run that op on the NPU, so it falls back to CPU for that op — which means the tensor must be copied from the NPU's memory to CPU memory and back, and that copy is pure overhead. Worse, the fallback *breaks the fusion chain*: the optimizer cannot fuse across the device boundary, so the conv before the unsupported op and the relu after it can no longer fuse together. A single unsupported op in a hot loop can cost you more than all your fusion wins combined, because it fragments the graph into many small device-pinned subgraphs with copies between them. The fix is to either replace the op with a supported equivalent, or to deliberately partition the graph so the unsupported op runs at a boundary where the copy is cheap. Reading the runtime's partitioning log — which subgraphs went to which device — is how you find this.

**What happens with dynamic shapes?** If you export with fully dynamic batch and sequence dimensions, constant folding cannot fold shape arithmetic (the shapes are symbolic), some fusions cannot fire (the kernel needs static dimensions), and layout assignment is more conservative. You keep paying for ops that would have folded with static shapes. The fix is to fix the dimensions you can — if your edge app always runs batch 1, export with batch 1, and if the sequence length is bounded, export with the bound — which lets the optimizer fold and fuse aggressively. There is a genuine trade-off here with flexibility, which [from model to deployable artifact](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact) discusses; on the edge, the inflexibility almost always pays for itself.

**What happens when the model is compute-bound, not memory-bound?** Then fusion helps less, because its main mechanism — removing memory traffic — is attacking the cost that was already hidden behind compute. You will still save kernel launches, but the big DRAM-traffic win is muted because the layer was bandwidth-rich to begin with. This is the case where hand-tuning a kernel or quantizing to use faster integer math units genuinely is the right move, and the roofline is how you tell the difference *before* you waste time. If your layer sits in the compute-bound region, do not expect 1.5x from fusion; expect single-digit percent, and look elsewhere.

## Case studies and real numbers

A few grounded references so these are not just my numbers.

**FlashAttention (Dao, Fu, Ermon, Rudra, Ré, 2022).** The canonical fusion result of the transformer era. By fusing the two attention matmuls and the softmax into one tiled kernel that never materializes the $N \times N$ score matrix in DRAM, FlashAttention reported 2–4x wall-clock speedups on attention and a memory footprint linear rather than quadratic in sequence length, with *exact* (not approximate) attention. The mechanism is precisely the one in this post — keep the intermediate in fast SRAM, never round-trip it — applied to the operator that dominates transformer inference. FlashAttention-2 (Dao, 2023) pushed the kernel-level scheduling further. For on-device transformers, a fused attention kernel is the difference between a model that fits the latency budget and one that does not.

**ONNX Runtime graph optimizations.** The ONNX Runtime documentation describes its three optimization tiers (basic, extended, layout/all) and the specific fusions it implements — conv-BN, conv-add, conv-activation, GELU fusion, attention fusion, embed-layer-norm fusion for transformers. On BERT-class models, ORT's transformer-specific fusions (attention and layer-norm fusion together) are reported to deliver multi-x CPU and GPU speedups over the unfused graph; the layer-norm and GELU fusions alone remove a large pile of memory-bound elementwise traffic. These are off-the-shelf, enabled by setting the optimization level — no model change.

**TensorRT layer and tensor fusion.** NVIDIA's TensorRT builder performs both vertical fusion (conv-bias-relu into one CBR kernel) and horizontal fusion (combining layers that share input dimensions), plus constant folding and precision-and-layout selection, as a core part of building an inference engine. NVIDIA's own materials attribute a substantial fraction of TensorRT's end-to-end speedup over a naive framework graph to these fusions, before any precision reduction. On Jetson targets this is the default path to extracting the hardware's throughput.

**The TVM / Halide line.** The research lineage behind automated graph and kernel optimization runs through Halide (Ragan-Kelley et al., 2013), which separated *what* to compute from *how* (schedule) it, and TVM (Chen et al., 2018), which brought operator fusion and learned schedule search to deep-learning graphs across diverse hardware backends. The graph-level fusion rules in today's runtimes are the productionized, rule-based descendants of that work; the search-based extensions (autotuning the schedule for the target) are covered in the compilers post. Understanding the rule-based core here is what makes the search-based layer make sense.

## What the compiler does for free versus what you must do

Here is the framing to walk away with, because it determines where you should spend your effort.

![A decision tree splitting each optimization into two branches: the compiler does fusion, BN folding, constant folding, dead-code elimination, and layout assignment automatically, while you must choose quantization and pruning, pick the target and execution provider, and verify numerical parity yourself.](/imgs/blogs/graph-level-optimization-8.png)

The graph-level transforms in this post — fusion, BN folding, constant folding, transpose cancellation, DCE, CSE, op simplification, layout assignment — are **automatic**. The runtime does them for you when you ask (one flag in ONNX Runtime, `Optimize.DEFAULT` in the TFLite converter, `torch.compile` in PyTorch). You do not implement them, you do not retrain, and they do not cost accuracy. They are the *last* step of the pipeline, applied to whatever model you hand the runtime, and they extract the remaining constant factor — typically a clean 1.5–2x with no downside. The most important practical implication: **always turn them on, and always measure before/after**, because it is the cheapest speedup you will ever get.

But — and this is the boundary that trips people up — the graph optimizer works with the model you give it. It will not quantize your model to int8 for you (that is a deliberate accuracy trade you must make and validate), it will not prune your channels (that changes the architecture), it will not distill a smaller student, and it will not choose your target hardware or execution provider. Those are *your* decisions, and they must come *first*, because they change what the graph optimizer then operates on. You quantize and prune to shrink the model and change its arithmetic; *then* the compiler fuses and folds whatever is left. The order is: pick the architecture (from [building blocks for efficient models](/blog/machine-learning/edge-ai/building-blocks-for-efficient-models)), quantize and prune (the heavy levers), export with the shapes and target fixed, and *finally* let the graph optimizer do its free pass on top. Graph optimization is the icing, not the cake — but it is the icing you should never skip, and it is the only layer that is genuinely free.

There is one more thing that is your job and that no compiler will do for you: **verify numerical parity**. The transforms are designed to preserve semantics up to floating-point rounding, and they almost always do — but a buggy custom op, an aggressive fusion on an edge case, or a layout assumption that does not hold can introduce a real discrepancy. After every optimization pass, run your evaluation set through the optimized model and confirm the accuracy and the output tensors match the reference within tolerance. This is cheap insurance and it has saved me more than once when a "free" optimization turned out to interact badly with a nonstandard op. The full discipline of this lives in [the edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle); for graph optimization specifically, the rule is simple: trust, but diff the outputs.

## When to reach for this (and when not to)

Graph optimization is unusual among the levers in this series because there is almost never a reason *not* to use it — but there are reasons it might not be *enough*, and knowing the difference saves you from chasing it past the point of diminishing returns.

**Always do it.** It is free, automatic, and accuracy-preserving. Turning on full optimization and measuring should be the first thing you do on any model, before you touch a heavier lever. If it closes your latency gap, you are done at zero accuracy cost.

**Do not expect it to save a compute-bound model.** Fusion's main win is removing memory traffic; if your hot layers are compute-bound (high arithmetic intensity, sitting under the compute roof), fusion gives you launch savings but not the big DRAM win. Reach for quantization (faster integer math) or a smaller architecture instead, and use the roofline to confirm which regime you are in before you decide.

**Do not let it be blocked by a custom op.** A single unsupported op in a hot path fragments the graph, forces device fallbacks and copies, and prevents fusion across it — which can cost more than all your fusion wins. If graph optimization underdelivers, the first suspect is an op that broke a chain. Replace it with a supported equivalent.

**Do not skip the shape work.** Dynamic shapes suppress constant folding and many fusions. If your edge app has fixed or bounded shapes, export with them fixed; the optimizer will reward you. The flexibility you give up is usually worth far less than the speed you gain.

**Do not assume it composes with quantization automatically.** Quantize first, then optimize the graph — and re-verify, because the interaction of fusion and quantized ops occasionally surfaces an edge case (a fused conv-BN that quantizes differently than the unfused pair). The order matters and the verification is not optional.

**Do reach past it when you have already won the free 1.5–2x and still miss target.** At that point you are out of free wins, and the honest move is the heavier levers — quantization, pruning, distillation, a smaller architecture — read off the accuracy–latency Pareto frontier. Graph optimization sets the floor; the levers move the floor.

## Key takeaways

- **The graph is the real program.** Your Python expresses intent; the inference runtime rewrites the resulting DAG before any kernel runs, and those rewrites are where the cheapest speedups live.
- **Fusion wins by removing memory traffic and launches, not FLOPs.** Merging conv-BN-relu keeps intermediates in registers instead of round-tripping them through DRAM, which on a bandwidth-poor edge device is the lever that matters — it directly raises arithmetic intensity.
- **BN folding is exact and free.** Because inference batchnorm is affine, it folds into the preceding conv's weights as $W' = sW$, $b' = s(b-\mu)+\beta$ with $s = \gamma/\sqrt{\sigma^2+\epsilon}$, deleting a node at zero runtime cost and enabling clean int8.
- **Constant folding pays once for what you would otherwise pay every inference.** Any subgraph with all-constant inputs is precomputed at build time — but it only fires when shapes are static enough to resolve.
- **Layout is a first-class edge optimization.** The right memory format (often NHWC on mobile NPUs) avoids cache-hostile strides and conversion ops; the wrong one can cost ~1.8x on a depthwise stack. Declare your target and let the runtime assign it.
- **The cleanup passes are the connective tissue.** DCE, CSE, transpose cancellation, and op simplification individually do little but collectively expose the big fusions, which is why optimizers run them first and to a fixed point.
- **It is the free, accuracy-preserving last step.** Always turn it on (`ORT_ENABLE_ALL`, `Optimize.DEFAULT`, `torch.compile`) and measure before/after honestly — warm up, report p50/p99, batch 1, watch for thermal throttling.
- **The compiler does the graph; you do everything before it.** Quantization, pruning, target choice, and parity verification are your job and come first; graph optimization is the icing you should never skip but cannot substitute for the cake.

## Further reading

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Dao, Fu, Ermon, Rudra, Ré (2022), and **FlashAttention-2** — Dao (2023). The defining modern fusion result; the IO-awareness framing is exactly the memory-traffic argument of this post.
- **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** — Chen et al. (2018), and **Halide** — Ragan-Kelley et al. (2013). The research lineage behind automated graph fusion and schedule search.
- **ONNX Runtime graph optimizations documentation** — the authoritative reference for the basic/extended/all tiers and the specific fusions ORT implements; read it alongside your own before/after op histograms.
- **TensorFlow Lite / LiteRT converter documentation** — how conversion-time constant folding, BN folding, and op fusion produce the deployed flatbuffer, and how delegates assign layout on device.
- **NVIDIA TensorRT developer guide** — layer and tensor fusion, constant folding, and precision/layout selection as the core of engine building on Jetson and discrete GPUs.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for the memory-bound-versus-compute-bound diagnosis that explains why fusion wins, [from model to deployable artifact](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact) for the export and shape decisions that unlock these transforms, [ML compilers and autotuning with TVM, MLIR, and XLA](/blog/machine-learning/edge-ai/ml-compilers-and-autotuning-tvm-mlir-xla) for the search-based extensions, and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for where graph optimization sits in the end-to-end flow.
