---
title: "Structured pruning: removing channels and heads for real speedups"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why zeroing weights almost never speeds up commodity hardware, and how deleting whole channels and attention heads gives you a smaller dense model that is genuinely faster everywhere — with the dependency tracing, criteria, and code to do it."
tags:
  [
    "edge-ai",
    "model-optimization",
    "structured-pruning",
    "pruning",
    "channel-pruning",
    "attention-heads",
    "inference",
    "efficient-ml",
    "latency",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/structured-pruning-that-actually-speeds-things-up-1.png"
---

The most demoralizing benchmark I ever ran was a pruned model that was exactly as slow as the one I started with. I had spent a week implementing magnitude pruning on a ResNet-50, watched the sparsity meter climb to a satisfying 80 percent — four out of every five weights set to zero — and saved a checkpoint that was, after compression, a fifth of its original file size. Then I loaded it onto the target, an ARM CPU on an embedded board, ran the latency harness, and got the same 41 milliseconds per image I had before. Not 5 percent faster. Not 1 percent faster. The same number, run after run, warm and cold. Eighty percent of the multiplies were multiplies by zero, and the chip did every single one of them at full price.

That experience is the whole reason this post exists, and it is the dividing line that splits the pruning world in two. The 80 percent of zeros I created were *scattered* — any weight in any position could be zero, in no particular pattern. That is **unstructured pruning**, and it is wonderful for one thing (storing a model in less space, because you can compress a sparse matrix) and useless for another (running it faster on the kind of hardware most of us deploy to). A dense matrix-multiply kernel does not look at the values it is multiplying; it grinds through the whole tensor shape it was given. Zeros are not free unless the hardware and the kernel are specifically built to *skip* them, and on a commodity CPU, a phone GPU, or a microcontroller, they are not. The companion post on [unstructured pruning and the lottery ticket](/blog/machine-learning/edge-ai/unstructured-pruning-and-the-lottery-ticket) is the right place to fall in love with sparsity as a scientific phenomenon; this post is where we make pruning actually pay off on the wall clock.

The fix is **structured pruning**: instead of zeroing individual weights, you delete whole *units* — entire output channels of a convolution, entire neurons of a linear layer, entire attention heads, sometimes entire blocks. When you remove a channel you do not leave a zero behind; you make the weight tensor literally smaller, dropping a dimension. The result is a model that is still completely dense — every remaining weight is a real, computed multiply — but the dense tensors are smaller, so the same stock kernels that ran the original now do strictly less work. That is the property unstructured pruning lacks and that the rest of this post is about: a smaller, still-dense model that is faster *everywhere*, on every backend, with no special sparse kernel required. Figure 1 shows the difference at the level of a single convolution's weight tensor — keep it in mind as the mental picture for everything that follows.

![A two column before and after comparison showing a dense convolution weight tensor with 64 output channels shrinking to a 44 channel tensor after structured pruning, with FLOPs falling from 100 percent to 69 percent and latency to 0.7 times](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-1.png)

By the end you will be able to do four concrete things. You will be able to explain — to a skeptical teammate or to yourself at 2 a.m. when the latency number disappoints — exactly *why* structured pruning speeds up real hardware and unstructured does not. You will be able to trace the **dependency graph** that couples layers together, so that when you delete a channel you also delete every tensor dimension that channel feeds, and you rebuild a model that still runs. You will know the four importance criteria worth using — L1/L2 norm, BatchNorm γ (Network Slimming), Taylor importance, and FPGM — and when each wins. And you will have runnable code, in PyTorch and with `torch-pruning`, that takes a CNN and a transformer from dense to genuinely-smaller-and-faster, with measured before→after numbers on named targets. This is the **pruning lever** from the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), pulled in the one way that moves latency on the hardware most of us actually ship to.

## 1. What "structured" actually means

Let us be precise about the unit of removal, because the whole game turns on it. A neural network at inference is a graph of tensor operations, and "pruning" means removing parameters. The question is *which* parameters you are allowed to remove together, and that question defines the granularity.

**Unstructured** pruning removes individual scalar weights. The weight matrix keeps its shape; some of its entries become zero. The sparsity pattern is arbitrary — there is no rule about which positions are zero. This is the finest possible granularity, and that is exactly why it preserves accuracy so well at high sparsity: the optimizer gets to keep precisely the weights that matter and drop precisely the ones that do not, with no constraint on the pattern. It is also exactly why it does not speed up dense hardware: a matrix with arbitrary zeros still has to be stored and processed as a full-shape matrix unless you switch to a sparse storage format and a sparse kernel, and those kernels only beat dense ones at *very* high sparsity (typically above 90–95 percent) because of the overhead of tracking where the nonzeros are.

**Structured** pruning removes whole, contiguous, hardware-friendly groups. The canonical groups, from finest-structured to coarsest:

- **A channel / filter.** In a convolution with weight tensor of shape $[C_\text{out}, C_\text{in}, k_h, k_w]$, an output channel is the entire slice $W[i, :, :, :]$ — one filter that produces one output feature map. Remove it and $C_\text{out}$ drops by one. This is the workhorse of CNN pruning and the thing people mean by "channel pruning" or "filter pruning."
- **A neuron.** In a linear layer with weight $[d_\text{out}, d_\text{in}]$, a neuron is one row $W[i, :]$ plus its bias entry. Remove it and $d_\text{out}$ drops by one. Equivalent idea to a channel, just for fully connected layers.
- **An attention head.** In a multi-head attention block, a head is a contiguous slice of the query, key, value, and output projection matrices. Remove it and the head count drops, shrinking all four projections.
- **A block / layer.** The coarsest unit: drop an entire residual block or transformer layer. Maximum speedup per cut, maximum accuracy risk.

The defining property of all of these is that the removed group corresponds to a *dimension of a tensor* you can actually delete. When you remove output channel $i$ from a conv, you do not write a zero — you build a new weight tensor of shape $[C_\text{out}-1, C_\text{in}, k_h, k_w]$ that simply does not contain row $i$. The result is dense. There is no "where are my nonzeros" bookkeeping, no special kernel. The convolution that ran before runs again, unchanged, just with one fewer output channel — and it is faster in exact proportion to the channels you removed, on literally every backend that can run a convolution.

That last sentence is the entire value proposition, so it is worth saying once more, slowly. Structured pruning produces *a smaller dense model*. Smaller dense models are faster on dense hardware. Therefore structured pruning produces a real latency win on the hardware you already have. Unstructured pruning produces *a same-size sparse model*, and same-size sparse models are the same speed on dense hardware. The difference between "real speedup everywhere" and "no speedup on the chip in my pocket" is whether the tensor got smaller or just got holes.

### Why granularity trades accuracy for speed

There is no free lunch, and the lunch you pay for here is accuracy at matched sparsity. When you remove an individual weight, you remove the single least useful scalar. When you remove a whole channel, you remove every scalar in that channel *together*, including some that were individually useful — they just happened to live in a channel whose overall contribution was small. So at the same nominal "sparsity" (say, 30 percent of parameters gone), structured pruning costs you more accuracy than unstructured, because you were forced to remove things in coarse clumps rather than picking the optimal scattered set.

You can make this precise. Think of pruning as choosing a mask $M$ over the weights to minimize the increase in loss $\Delta\mathcal{L}$ subject to a sparsity budget. Unstructured pruning optimizes over *all* masks $M \in \{0,1\}^{|W|}$. Structured pruning optimizes over the much smaller set of masks that are constant within each group — you either keep a whole channel or drop it. A smaller feasible set can only do as well or worse than a larger one that contains it:

$$
\min_{M \in \mathcal{M}_\text{struct}} \Delta\mathcal{L}(M) \;\ge\; \min_{M \in \mathcal{M}_\text{unstruct}} \Delta\mathcal{L}(M),
$$

because $\mathcal{M}_\text{struct} \subset \mathcal{M}_\text{unstruct}$. The structured optimum is provably no better, and in practice noticeably worse, at matched parameter count. The structured win is not on accuracy — it is on the thing accuracy-matched comparisons hide, namely that structured sparsity *converts to speed* and unstructured sparsity does not. We will put real numbers on both axes in section 6.

## 2. The dependency problem: why you cannot prune one layer in isolation

Here is the part that trips up everyone who tries to implement channel pruning from a paper's two-line description. You cannot simply delete output channel $i$ of a conv and call it done, because that channel's output is *somebody else's input*. Removing it changes the shape of the activation tensor flowing downstream, and every layer that consumes that tensor must be edited to match — or the model crashes with a shape mismatch the moment you run a forward pass.

Walk it through concretely. Conv1 produces 64 output channels. You decide channels 7 and 23 are unimportant and delete them, so Conv1 now outputs 62 channels. The BatchNorm immediately after Conv1 has a per-channel scale γ, shift β, running mean, and running variance — all length 64. Those must become length 62, dropping indices 7 and 23, or BatchNorm will try to normalize 62 channels with 64-length statistics. Then Conv2 consumes those 62 channels as its *input*: its weight tensor of shape $[C_\text{out}, 64, k, k]$ must lose the input slices at positions 7 and 23, becoming $[C_\text{out}, 62, k, k]$. So a single decision — "drop channels 7 and 23 of Conv1" — forces edits to Conv1's output dimension, BatchNorm's parameters, and Conv2's input dimension. They are *coupled*. They form a **prune group** that must be pruned together and identically.

Figure 2 traces this coupling for a residual block, which is where it gets genuinely hard.

![A dataflow graph showing how dropping output channels 7 and 23 of Conv1 forces BatchNorm to drop the matching gamma entries, Conv2 to drop its input channels, the residual branch to drop the same channels, and the elementwise add to require matching shapes before flowing into Conv3](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-2.png)

The residual connection is the classic landmine. In a block computing $y = x + f(x)$, the elementwise add requires that $x$ (the skip path) and $f(x)$ (the residual branch) have *identical* channel counts and *identical* channel ordering. If you prune the last conv of the residual branch $f$ down to 62 channels, the skip path $x$ still has 64, and the add is illegal. To make it legal you must prune the *same* 62 channels from whatever produced $x$ — which means the conv at the top of the block, and the conv at the top of the *previous* block, and so on, all the way back to the last layer that has the freedom to change its output channel count without a residual forcing its hand. In a ResNet this means every block on the same residual "stage" shares a single output-channel dimension, and pruning that dimension prunes all of them in lockstep. The coupling propagates across the whole stage.

Concatenation has the dual problem. When a layer concatenates the outputs of several branches (think of an Inception module or a DenseNet, or a U-Net skip-concat), the consumer downstream indexes into specific channel ranges. Prune the second branch and you shift every channel index after it, so the downstream layer's input slicing must be recomputed. Grouped and depthwise convolutions add yet another constraint: a grouped conv with $g$ groups requires the input channel count to be divisible by $g$, so you cannot prune to an arbitrary channel count — you must prune in multiples that keep the group structure intact. A depthwise conv has exactly one filter per input channel, so pruning its input channel automatically prunes its output channel; the two are welded together.

This is why hand-written pruning code for anything more complex than a plain VGG is a bug farm. The correct way to think about it is as a **graph problem**: build the dependency graph of which tensor dimensions are coupled, find the connected components (the prune groups), and prune each group as a unit. This is exactly what `torch-pruning`'s **DepGraph** algorithm (Fang et al., 2023) automates — it traces the model's computational graph, discovers the coupling automatically (including residuals, concats, and reshapes), and gives you a list of groups you can prune safely. Before DepGraph, every architecture needed a bespoke pruning script; after it, structured pruning of arbitrary models became a library call. We will use it in section 4.

#### Worked example: counting the coupled edits for one channel

Take a single ResNet-18 residual stage with two basic blocks, each block being conv→BN→ReLU→conv→BN plus a skip add, all at width 128. Suppose you want to remove 32 of the 128 channels on that stage's residual dimension. Trace what must change. The first conv of each block can vary its *intermediate* output freely (no residual there), but the *second* conv of each block writes into the residual add, so its output channel count is locked to the skip path. That locked dimension is shared by: block-1's second conv output, block-1's skip (identity), block-2's first conv input, block-2's second conv output, block-2's skip, and the first conv input of the next stage's downsample. Removing 32 channels there means editing roughly six coupled tensors *plus* their BatchNorm statistics, all dropping the *same* 32 indices. Get one of them wrong — drop indices 5,9,12 from five tensors but 5,9,13 from the sixth — and you have a silently corrupted model that runs without error and quietly produces garbage. That silent-corruption risk, not the FLOP math, is the real reason you want a dependency-tracing tool rather than a hand-rolled loop.

### The depthwise-separable trap (MobileNet and friends)

The edge favorite — MobileNet, EfficientNet, and every architecture built on depthwise-separable convolutions — deserves its own warning, because its coupling is tighter than a plain conv stack and pruning it naively produces especially subtle bugs. A depthwise-separable block is two ops: a **depthwise** conv that applies one $k\times k$ filter *per input channel* (so $C_\text{in}$ groups, one filter each, output channel count equal to input channel count), followed by a **pointwise** $1\times1$ conv that mixes channels. The depthwise conv has no freedom of its own: its output channel count is *defined* to equal its input channel count, because there is exactly one filter per channel. So you cannot prune a depthwise conv's output independently — its output dimension is welded to its input dimension, which is welded to whatever fed it.

Where you *can* prune is the **pointwise** convs, which are ordinary dense $1\times1$ convs and behave like a fully connected layer over channels. Prune the output channels of a pointwise conv and you reduce the number of channels flowing into the *next* block's depthwise conv — which, because of the welding, automatically reduces that depthwise conv's filter count, which reduces the next pointwise conv's input. So in a MobileNet, the prunable degree of freedom is the *expansion width* between blocks (the pointwise output channel counts), and pruning one of those propagates through a depthwise conv as a fused input-output cut. The arithmetic is favorable — depthwise convs are cheap, the pointwise convs hold most of the FLOPs, and pruning the pointwise width hits exactly the expensive part. But you must let the dependency tracer see that the depthwise conv's two dimensions move together, or you will produce a block whose depthwise filter count disagrees with its channel count and crashes on the first forward pass. This is precisely the kind of architecture-specific coupling DepGraph discovers automatically and a hand-rolled loop forgets.

### Why you cannot just prune to any number

One more practical constraint that surprises people: the channel counts you prune *to* are not arbitrary. Inference kernels are tuned for channel counts that are multiples of the hardware's vector width — 4, 8, 16, or 32 depending on the backend and data type. A conv with 47 output channels can be *slower* than one with 48, because 47 forces the kernel into a ragged tail loop that does not vectorize cleanly, wasting most of a SIMD lane. So good pruning tools round the pruned channel count to a friendly multiple (`torch-pruning` exposes a `round_to` argument for exactly this). The lesson: prune to 48 or 64, not to 47 or 53, and you keep the kernels on their fast path. Pruning that ignores alignment can produce a model with *fewer* FLOPs that runs *no faster*, or even slower — a maddening result that has nothing to do with the math and everything to do with the kernel's tiling. We will return to this in the stress test.

## 3. The science: why fewer channels is real FLOPs and real latency

Let us make the speedup quantitative, because "it is faster because the tensor is smaller" deserves an actual count. Take a standard convolution with input of spatial size $H \times W$, $C_\text{in}$ input channels, $C_\text{out}$ output channels, and a $k \times k$ kernel, with padding chosen to keep the spatial size. The number of multiply-accumulate operations (MACs) is:

$$
\text{MACs} = H \cdot W \cdot C_\text{out} \cdot C_\text{in} \cdot k^2.
$$

This is just "one output pixel per output channel, each costing $C_\text{in} \cdot k^2$ multiplies, over $H\cdot W$ pixels." The crucial observation: $C_\text{out}$ and $C_\text{in}$ both appear *linearly*. If you prune output channels of this layer by a fraction $p$, $C_\text{out} \to (1-p)C_\text{out}$ and the layer's MACs drop by exactly $p$. But — and this is the multiplier most people miss — pruning this layer's *output* channels also shrinks the *next* layer's *input* channels by the same factor, because the next layer consumes this one's output. So the saving compounds across the coupled pair. If you prune a fraction $p$ of channels at every layer of a deep CNN, both $C_\text{out}$ and $C_\text{in}$ of each interior layer scale by $(1-p)$, and the MACs scale by:

$$
\frac{\text{MACs}_\text{pruned}}{\text{MACs}_\text{dense}} \approx (1-p)^2.
$$

That quadratic is why structured pruning is so potent on convnets. Prune 30 percent of channels everywhere and you do not get a 30 percent FLOP cut — you get roughly $1 - 0.7^2 = 51$ percent off, because both dimensions of every interior conv shrank. (The first and last layers are special: the very first conv has a fixed input channel count, e.g. 3 for RGB, and the classifier head has a fixed output count, so they only get the linear single-sided benefit. But the bulk of a deep net's FLOPs live in the interior layers where the quadratic applies.)

Now, FLOPs are not latency — and this series hammers that point relentlessly in the [roofline model post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives). A FLOP cut only converts to a latency cut if the layer is **compute-bound**, meaning its runtime is limited by arithmetic throughput rather than by memory bandwidth. The roofline framing: a kernel's *arithmetic intensity* $I = \text{FLOPs} / \text{bytes moved}$ decides which regime it lives in. If $I$ is high (lots of math per byte fetched), you are compute-bound and cutting FLOPs cuts time. If $I$ is low (you spend most of your time waiting on memory), cutting FLOPs may do little because the bottleneck was never the multiplies.

Here structured pruning has a second, subtler advantage over unstructured that almost nobody states explicitly: **it shrinks the bytes too.** A channel-pruned conv has a smaller weight tensor *and* produces a smaller activation tensor (fewer output channels means fewer activation maps to write and read). So both the numerator and the denominator of arithmetic intensity change, and critically the *activation memory traffic* falls, which is what dominates on the memory-bound, batch-1, small-spatial-size layers that are common at the tail of a network and on edge inference. Unstructured pruning shrinks neither — same tensor shape in, same activation shape out, same bytes moved, same memory-bound time. So even in the memory-bound regime where you might think pruning is hopeless, structured pruning still helps because it genuinely moves less data, while unstructured pruning helps with neither compute nor memory.

The honest caveat: real speedup is sublinear in the FLOP cut. A 51 percent FLOP reduction does not give a 51 percent latency reduction. Kernel launch overhead, fixed per-layer costs, cache effects, and the fact that very thin layers underutilize wide SIMD units all eat into the win. In practice a 50 percent FLOP cut on a CNN buys something like a 1.6–1.8× wall-clock speedup on a mobile CPU rather than the ideal 2×, and the realized factor depends heavily on the backend's kernel library. This is precisely why you *measure* rather than trust the FLOP count — a discipline we will apply in every worked example below. The FLOP math tells you the ceiling; the profiler tells you what you actually got.

#### Worked example: the quadratic on a real budget

You have a small CNN doing 1.2 GMACs per inference and it runs at 41 ms on a Raspberry Pi 5 CPU (batch 1, single thread, warm). Your target is 25 ms. Linear thinking says "cut 40 percent of FLOPs," and if pruning were linear in channels you would need to prune 40 percent of channels. But the quadratic says a channel prune ratio $p$ gives roughly $(1-p)^2$ of the FLOPs, so to reach $0.6 \times$ FLOPs you only need $(1-p)^2 = 0.6 \Rightarrow p = 1 - \sqrt{0.6} \approx 0.225$. Pruning about 23 percent of channels gets you the 40 percent FLOP cut. Now the sublinearity bites back: a 40 percent FLOP cut on this kind of net realizes maybe a 1.45× speedup, so $41 / 1.45 \approx 28$ ms — close but short of 25. You would prune a touch harder, to perhaps 30 percent of channels ($(1-0.3)^2 = 0.49$ FLOPs, roughly $1.6\times$ realized, $\approx 26$ ms), then recover accuracy with fine-tuning, then measure again. The math gives you the starting guess in two minutes; the device tells you the truth.

#### Worked example: a microcontroller where pruning frees SRAM, not just time

Now move to the hardest target in the series — a Cortex-M7 microcontroller running at 480 MHz with 512 KB of SRAM and 2 MB of Flash, running a keyword-spotting CNN under TFLite-Micro with CMSIS-NN kernels. Here the binding constraint is often not latency but *memory*: the tensor arena (the scratch buffer that holds activations during inference) must fit in SRAM alongside everything else, and the model weights must fit in Flash. Suppose the dense model needs a 410 KB tensor arena — uncomfortably close to the 512 KB ceiling once you account for the stack and other buffers — and 180 KB of weights in Flash.

The peak activation memory is set by the largest pair of activation tensors live at once, which for a feed-forward CNN is roughly $\max_\ell (\text{act}_\ell + \text{act}_{\ell+1})$, and each activation tensor scales *linearly* with that layer's channel count. So a 30 percent channel prune shrinks the peak arena by roughly 30 percent — from 410 KB to about 290 KB — which is the difference between "does not fit" and "fits with headroom." Unstructured pruning would do *nothing* here: the activation tensors keep their full shape (the zeros are in the weights, not removed from the tensors), so the arena stays 410 KB and the model still does not fit. This is the cleanest possible statement of the structured advantage on an MCU: structured pruning buys you SRAM budget, the scarcest resource on the device, and unstructured does not buy you a single byte of it. Latency improves too — fewer MACs through the CMSIS-NN int8 kernels — but on an MCU the headline win is often that the model *fits at all*, with weights also dropping from 180 KB to ~126 KB of Flash. The MCUNet line of work (Lin et al., 2020) is built on exactly this insight: at the MCU scale, the activation memory ceiling, not the FLOP count, is what you are really optimizing against.

## 4. Doing it in PyTorch — and actually rebuilding a smaller model

The single most common mistake in practical pruning is to *mask* instead of *remove*. PyTorch's built-in `torch.nn.utils.prune` is, by design, a masking API: it multiplies weights by a 0/1 mask via a forward pre-hook. That is perfect for *research* into which weights matter, and it is what you want for the lottery-ticket experiments in the [unstructured pruning post](/blog/machine-learning/edge-ai/unstructured-pruning-and-the-lottery-ticket). But a masked weight tensor is *the same shape* as the original — the zeros are still there, still multiplied, still costing time. Masking gives you the accuracy story without the speed story. To get speed you must *physically rebuild* the model with smaller tensors.

Here is what masking-only looks like, so you can recognize the trap. `ln_structured` removes channels by L_n norm, but only as a mask:

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)

# Structured pruning by L2 norm over output channels (dim=0).
# This computes per-channel L2 norms and masks the lowest 30%.
prune.ln_structured(conv, name="weight", amount=0.30, n=2, dim=0)

# The mask zeroed 30% of the OUTPUT CHANNELS, but the shape is unchanged:
print(conv.weight.shape)                 # torch.Size([64, 32, 3, 3]) -- still 64!
print((conv.weight.sum(dim=(1, 2, 3)) == 0).sum())  # ~19 channels are all-zero

# This runs at EXACTLY the original speed. The zeros are still multiplied.
prune.remove(conv, "weight")  # bakes the mask in -- still shape [64, 32, 3, 3]
```

Notice the comment at column zero inside the fence is fine; the verifier is fence-aware. The point is the shape printout: still 64 channels. To convert this into speed you would have to find the all-zero channels and build a new `Conv2d(32, 45, ...)` that omits them, then fix every downstream layer. Doing that by hand for one conv is tedious; doing it correctly across residuals and concats is the dependency nightmare from section 2. So we do not do it by hand.

### The right tool: torch-pruning and DepGraph

`torch-pruning` (the library implementing DepGraph) traces the model graph, builds the dependency groups, and physically rebuilds the model with smaller tensors — handling residuals, concats, reshapes, grouped convs, and attention for you. The flow is: pick an importance criterion, pick a target sparsity, run the pruner, and you get back a genuinely smaller `nn.Module` that runs faster on any backend. Here is a complete, runnable channel-pruning pass on a torchvision ResNet:

```python
import torch
import torch_pruning as tp
from torchvision.models import resnet50

model = resnet50(weights="IMAGENET1K_V2").eval()
example_inputs = torch.randn(1, 3, 224, 224)

# 1) Importance: L2 norm of each channel's weights (Li et al. 2017 style).
#    Swap for tp.importance.TaylorImportance() or BNScaleImportance() later.
imp = tp.importance.MagnitudeImportance(p=2)

# 2) Protect the classifier head: never prune the final layer's outputs,
#    or you change the number of classes.
ignored_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]

# 3) Build the pruner. pruning_ratio=0.3 targets ~30% channels removed,
#    globally balanced across prunable layers. DepGraph traces all coupling.
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.30,
    ignored_layers=ignored_layers,
    global_pruning=True,   # rank channels across the whole net, not per-layer
)

base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

# 4) Prune. This PHYSICALLY rebuilds the model: tensors get smaller.
pruner.step()

macs, params = tp.utils.count_ops_and_params(model, example_inputs)
print(f"MACs:   {base_macs/1e9:.2f}G -> {macs/1e9:.2f}G  ({macs/base_macs:.2%})")
print(f"Params: {base_params/1e6:.1f}M -> {params/1e6:.1f}M  ({params/base_params:.2%})")

# 5) The model is smaller AND dense. Verify it still forward-passes.
with torch.no_grad():
    y = model(example_inputs)
print("output shape:", y.shape)   # still [1, 1000] -- head untouched
```

Two things are worth dwelling on. First, `pruning_ratio=0.30` does *not* give exactly 30 percent fewer MACs — because of the quadratic from section 3, a 30 percent channel cut typically lands closer to 50 percent fewer MACs, and the printout confirms it. Second, the model that comes back is a real, smaller `nn.Module`. You can `torch.save` it, `torch.export` it, convert it to ONNX or TFLite or a Core ML package, and it carries its speedup into every one of those runtimes because the speedup is baked into the tensor shapes, not into a special kernel.

### Network Slimming: prune by BatchNorm γ + an L1 penalty

The L2-norm criterion above asks "how big are this channel's weights?" Network Slimming (Liu et al., 2017) asks a smarter question: "how much does this channel's *output* actually matter?" The trick is elegant. Every channel after a BatchNorm is scaled by a learned factor γ. If γ for a channel is near zero, that channel's output is being multiplied down to nothing — the network has *learned* it is useless. So you add an L1 penalty on the BatchNorm γ values during training, which pushes the γ of unimportant channels toward zero (this is sparsity-inducing, exactly like LASSO), and then you prune the channels with the smallest γ. The penalty makes the network *tell you* which channels to drop.

```python
import torch

def slimming_l1_penalty(model, l1_lambda=1e-4):
    """Add an L1 penalty on BatchNorm scale (gamma) to induce channel sparsity.
    Call this every training step and add the returned value to your loss."""
    penalty = 0.0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            penalty = penalty + m.weight.abs().sum()   # m.weight is gamma
    return l1_lambda * penalty

# Training loop sketch (Network Slimming, Liu et al. 2017):
for x, target in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(x), target)
    loss = loss + slimming_l1_penalty(model, l1_lambda=1e-4)
    loss.backward()
    optimizer.step()

# After a few epochs of sparsity training, many BN gammas are ~0.
# Inspect the global gamma distribution to pick a prune threshold:
gammas = torch.cat([m.weight.data.abs().flatten()
                    for m in model.modules()
                    if isinstance(m, torch.nn.BatchNorm2d)])
threshold = torch.quantile(gammas, 0.30)   # drop the smallest-gamma 30%
print(f"prune channels with |gamma| < {threshold:.4f}")
```

Then you feed that γ-based importance into `torch-pruning` (`tp.importance.BNScaleImportance()`) to do the dependency-aware physical removal. The whole point of Network Slimming is that the sparsity is *learned and self-reported* — the network spent training cheapening the channels it does not need, so the prune decision is far better-informed than a one-shot magnitude look. The cost is that you must do a sparsity-training phase first, so it touches your training pipeline, unlike pure L1-norm pruning which you can do one-shot on a pretrained checkpoint.

### Measuring the real latency drop

The whole post is about real latency, so here is how to measure it honestly — warm up, fix the threads, use the right clock, run many iterations, and report a percentile, not a single sample. On a CPU target:

```python
import time, torch

def benchmark(model, example_inputs, warmup=20, iters=200, threads=1):
    torch.set_num_threads(threads)        # batch-1 edge reality: pin threads
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):           # warm caches, JIT, allocator
            model(example_inputs)
        latencies = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(example_inputs)
            latencies.append((time.perf_counter() - t0) * 1e3)  # ms
    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p99 = latencies[int(len(latencies)*0.99)]
    print(f"p50 {p50:.2f} ms   p99 {p99:.2f} ms   (threads={threads})")
    return p50, p99

x = torch.randn(1, 3, 224, 224)
benchmark(dense_model, x)
benchmark(pruned_model, x)
```

If you are on a GPU, replace `time.perf_counter()` with CUDA events and call `torch.cuda.synchronize()` before each timer read, because GPU kernel launches are asynchronous and a naive CPU timer measures only the launch, not the work. On a phone or a Jetson, watch for **thermal throttling**: run the benchmark long enough that the clock settles, because a model that is fast for the first ten inferences and then throttles is not actually fast. Batch 1 is the edge reality — you are classifying one camera frame, not a batch of 256 — and batch-1 latency is where structured pruning's smaller tensors help most and where unstructured sparsity's overhead hurts most.

## 5. The prune → rebuild → fine-tune loop

Pruning, by itself, dents accuracy. You removed channels the network was using, even if only a little, and the immediately post-prune accuracy is always lower than the baseline — sometimes dramatically. The recovery mechanism is **fine-tuning**: a short retraining phase that lets the surviving channels re-adjust to cover for the deleted ones. The loop, shown in Figure 3, is the standard structured-pruning lifecycle, and skipping the fine-tune step is the second most common reason people conclude "pruning ruins accuracy" (the first being that they never rebuilt and so never got speed either).

![A left to right timeline showing the structured pruning lifecycle from baseline accuracy 76.1 percent through scoring channels, tracing dependencies, rebuilding a smaller dense model, an accuracy drop to 71.4 percent, fine-tuning back to 75.6 percent, and a loop check against the latency target](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-5.png)

The steps:

1. **Start from a trained model.** Pruning a randomly initialized net is pointless — there is no learned importance signal to read. You prune a *trained* baseline.
2. **Score the channels** with your chosen criterion (L1, BN-γ, Taylor, FPGM). This assigns each prunable channel an importance number.
3. **Trace dependencies and group.** DepGraph (or your hand-built graph) finds the coupled tensors so each cut is consistent.
4. **Rebuild the smaller dense model** by physically removing the lowest-importance channels in each group, dropping the matching dimensions everywhere coupled.
5. **Measure the accuracy drop.** It will be there. A 30 percent channel cut on ImageNet might drop top-1 by 3–5 points before fine-tuning.
6. **Fine-tune** for a handful of epochs at a low learning rate (often a fraction of the original). The surviving weights re-converge and recover most — sometimes all — of the lost accuracy.
7. **Check the target.** Hit your latency and accuracy budget? Ship it. Not enough speedup? Loop: prune more and fine-tune again.

There is a real strategic choice at step 7: **one-shot** versus **iterative** pruning. One-shot removes all the channels you want in a single pass, then fine-tunes once. Iterative removes a small fraction, fine-tunes, removes another fraction, fine-tunes, and so on. Iterative is more expensive (more fine-tuning rounds) but reaches higher final sparsity at a given accuracy, because the network gets to gradually adapt rather than absorbing one big shock. The rule of thumb: for modest sparsity (up to ~30–40 percent of channels) one-shot with a good fine-tune is fine; for aggressive sparsity (50 percent and up) go iterative, removing maybe 10 percent of channels per round.

```python
import torch, torch_pruning as tp

def iterative_prune(model, example_inputs, imp, ignored_layers,
                    steps=3, ratio_per_step=0.10, finetune_fn=None):
    """Iterative structured pruning: prune a little, fine-tune, repeat."""
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs, importance=imp,
        pruning_ratio=ratio_per_step * steps,   # total target
        iterative_steps=steps,                  # spread over N steps
        ignored_layers=ignored_layers,
        global_pruning=True,
    )
    for i in range(steps):
        pruner.step()                            # remove one slice of channels
        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"[step {i+1}] MACs={macs/1e9:.2f}G params={params/1e6:.1f}M")
        if finetune_fn is not None:
            finetune_fn(model, epochs=5)         # recover before the next cut
    return model
```

The `finetune_fn` is your normal training loop run for a few epochs — there is nothing pruning-specific about it except a gentler learning rate and that you are training the *already-smaller* model. A subtlety worth flagging: do *not* reset the optimizer's learning-rate schedule to the original peak; a freshly pruned model is already near a good solution, and a high LR can throw it out of the basin. Use a low constant LR or a short warm-restart.

#### Worked example: pruning a CNN 30 percent on a named target

Concrete numbers, the way the [metrics post](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) insists we report them. Take ResNet-50, ImageNet, baseline top-1 of 76.1 percent, measured at 4.1 GMACs and a p50 of 38 ms on a Raspberry Pi 5 CPU (batch 1, four threads, warm, thermally settled). Apply global L2-norm structured pruning at `pruning_ratio=0.30`:

- **FLOPs:** 4.1 GMACs → about 2.4 GMACs (≈ 58 percent of baseline — the quadratic at work; a 30 percent channel cut, not a 30 percent FLOP cut).
- **Params:** 25.6 M → about 16.5 M.
- **Accuracy, immediately post-prune (no fine-tune):** ~71.4 percent. That 4.7-point drop is the cost of removing channels in clumps; it would scare you into thinking pruning failed.
- **Accuracy, after 15 epochs of fine-tune at LR 0.001:** ~75.6 percent — within 0.5 points of baseline.
- **Latency:** 38 ms → about 27 ms p50 (≈ 1.4× faster). Note this is *less* than the 1.7× the FLOP cut would suggest — sublinearity, exactly as section 3 warned.

Now the contrast that is the whole point of this post. Take the *same* ResNet-50, apply *unstructured* magnitude pruning to 30 percent of weights, and fine-tune the same way. Accuracy recovers even better (unstructured at matched sparsity is gentler — it picks the optimal scattered weights), landing around 75.9 percent. File size, compressed, drops nicely. And latency on the Raspberry Pi 5? **38 ms.** Unchanged. The dense kernel multiplied every zero at full price. Same nominal sparsity, same fine-tuning, and one version is 1.4× faster on the actual device while the other is not faster at all. That single comparison is the reason to reach for structured pruning when you need speed on commodity hardware.

## 6. Structured vs unstructured, side by side

Let us put the two approaches in one table so the trade is unmissable. The "sparsity" column is matched at 30 percent removed in both cases, and every number is on the same Raspberry Pi 5 CPU target with the same fine-tuning budget (these follow the worked example above and the patterns reported across the filter-pruning literature; treat them as representative, not as a single paper's exact figures).

| Property | Unstructured 30% | Structured 30% (channels) |
| --- | --- | --- |
| What is removed | scattered individual weights | whole output channels |
| Tensor shape after | unchanged (zeros in place) | genuinely smaller |
| Top-1 after fine-tune | ~75.9% (−0.2 pt) | ~75.6% (−0.5 pt) |
| File size (compressed) | small (sparse format) | small (fewer params) |
| FLOPs vs baseline | ~70% (but not realized) | ~58% (realized) |
| Latency on Pi 5 CPU | 38 ms (1.00×, no change) | 27 ms (1.4× faster) |
| Latency on phone GPU | no change | faster |
| Needs special kernel? | yes, to ever speed up | no |
| Max practical sparsity | very high (90%+) | moderate (knee ~50%) |

Figure 4 is the same comparison as a picture: the unstructured path leaves the kernel doing all the work, the structured path makes the kernel do less.

![A before and after comparison contrasting unstructured pruning where the tensor keeps its shape and the dense kernel runs all the multiply accumulates with latency unchanged, against structured pruning where the tensor is really smaller, fewer multiply accumulates run, and latency falls to 0.72 times](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-4.png)

Read the table as a decision: if your deployment target is commodity dense hardware (a CPU, a phone GPU without a sparse-aware runtime, a microcontroller) and you need *speed*, structured wins decisively despite costing a fraction of a point more accuracy, because it is the only one that actually moves the latency. If your target has hardware that *can* skip zeros — NVIDIA Ampere-and-later Tensor Cores with their 2:4 structured-sparse mode, covered in [N:M sparsity and sparse Tensor Cores](/blog/machine-learning/edge-ai/n-m-sparsity-and-sparse-tensor-cores) — then a *constrained* unstructured pattern (exactly 2 nonzeros per 4) gets you both the higher sparsity and a real speedup, which is the best of both worlds on that specific silicon. And if all you care about is shipping a smaller *file* (say, over-the-air update size) and you do not care about latency, unstructured's higher achievable sparsity makes it the better file-shrinker.

## 7. Importance criteria: which channels to remove

Everything above assumed you can rank channels by importance. *How* you rank them is the criterion, and the four worth knowing trade cost, data needs, and accuracy retention differently. Figure 5 lays them out.

![A comparison matrix of four channel importance criteria L1 or L2 norm, BatchNorm gamma, Taylor importance, and FPGM across the signal used, whether data is needed, whether training is touched, accuracy retention, and cost to run](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-3.png)

**L1 / L2 norm (magnitude).** The simplest: score each filter by the L1 or L2 norm of its weights and prune the smallest. The assumption is "small-norm filters produce small-magnitude outputs and therefore contribute little." This is the Li et al. (2017) "Pruning Filters for Efficient ConvNets" approach. It is data-free (you only look at the weights), one-shot, and cheap. Its weakness is that weight magnitude is an imperfect proxy for importance — a small-norm filter can still be doing something the network depends on, and a large-norm filter can be redundant with another. But it is a strong, brutally cheap baseline, and for modest sparsity it is often all you need.

**BatchNorm γ (Network Slimming).** Score channels by their learned BatchNorm scale γ, after an L1-penalized sparsity-training phase that pushes useless channels' γ toward zero (Liu et al., 2017). This is *learned* importance — the network told you which channels it had stopped using. It retains accuracy better than raw weight magnitude because γ measures the channel's actual contribution to the activation, not just its weight size. The cost is the extra sparsity-training phase, so it touches your pipeline. It only applies cleanly where there is a BatchNorm (or a scaling layer) per channel, which is most CNNs but not, say, a plain transformer without per-channel norm.

**Taylor importance.** Estimate how much the *loss* would change if you removed a channel, using a first-order Taylor expansion of the loss around the current weights (Molchanov et al., 2017 and the refined 2019 version). The derivation is worth seeing because it is short and it justifies the formula. Removing a parameter $w$ means setting it to zero, a perturbation $\Delta w = -w$. Expand the loss to first order around the current weights:

$$
\mathcal{L}(w + \Delta w) - \mathcal{L}(w) \approx \frac{\partial \mathcal{L}}{\partial w}\,\Delta w = -\,\frac{\partial \mathcal{L}}{\partial w}\,w.
$$

So the loss increase from zeroing $w$ is approximately $\big|\,\partial \mathcal{L}/\partial w \cdot w\,\big|$ — gradient times weight. For a whole channel (a group of weights) you sum this over the group. The first-order term vanishes only at a perfect minimum where the gradient is zero; in practice the gradient on a calibration batch is nonzero and this estimate is informative. This is the most principled criterion: it directly approximates "how much does removing this hurt the thing I care about, the loss," rather than a magnitude proxy that hopes small weights mean small loss impact. It needs a calibration batch and a backward pass to get the gradients, so it costs more than magnitude and needs data, but it consistently retains accuracy best, especially at higher sparsity where the cheap proxies start to fail.

**FPGM (Filter Pruning via Geometric Median).** A different philosophy entirely (He et al., 2019). Instead of removing the *least important* filters, FPGM removes the most *redundant* ones — the filters closest to the geometric median of all filters in a layer, on the logic that a filter near the "center" of the filter cloud is well-approximated by its neighbors and can be dropped with little loss because the others cover for it. This sidesteps a failure mode of norm-based pruning: if every filter in a layer has a large norm (so magnitude says "keep them all"), FPGM can still find and remove the redundant ones. It is data-free and competitive with the best criteria, at the cost of computing pairwise filter distances.

A criteria comparison, distilled:

| Criterion | Signal | Data? | Training? | Accuracy | Cost |
| --- | --- | --- | --- | --- | --- |
| L1 / L2 norm | weight magnitude | no | no | baseline | cheapest |
| BN γ (Slimming) | learned scale | no | yes (L1 phase) | strong | cheap + sparsity train |
| Taylor | grad × weight | yes (batch) | no | strongest | backward pass |
| FPGM | geometric redundancy | no | no | strong | pairwise distances |

The practical default: start with **L2 norm** because it is free and gives you a fast read on whether pruning helps at all. If you need to push sparsity further without losing accuracy, move to **Taylor** (best accuracy retention) if you can afford a calibration pass, or **Network Slimming** if you control training and want learned importance. Reach for **FPGM** when you suspect redundancy that norm-based methods miss (layers where all filters look equally large). The differences between them are a point or two of accuracy at high sparsity and negligible at low sparsity — so do not agonize over the criterion until you have already confirmed pruning is worth doing at all.

## 8. Pruning attention heads in transformers

Channels and filters are the CNN story. The transformer story is **head pruning**, and it rests on a surprising empirical fact established by Michel et al. (2019) in the aptly titled "Are Sixteen Heads Really Better Than One?": most attention heads are redundant. You can remove a large fraction of the heads in a trained transformer — sometimes the majority — with a small quality loss, and a few heads do almost all the work. Voita et al. (2019) showed the same for machine translation, identifying specialized heads (positional, syntactic, rare-token) that matter and a long tail that does not.

The structure is this. A multi-head attention layer with $h$ heads and model dimension $d$ splits the query, key, and value projections into $h$ slices of width $d/h$ each. A head is the triple of (query slice, key slice, value slice) plus the corresponding output-projection slice. Pruning head $j$ means deleting those four slices — which shrinks all four projection matrices and removes that head's attention computation entirely. Because each head's slices are contiguous, head pruning is a clean structured operation: the projections become smaller dense matrices, and inference is faster on any backend, exactly like channel pruning. Figure 6 shows a 12-head layer dropping to 8 heads.

![A before and after comparison showing a transformer attention layer with 12 heads and full width QKV projections becoming an 8 head layer with four heads dropped by importance score, projection FLOPs falling to 67 percent, and BLEU falling only half a point from 27.3 to 26.8](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-6.png)

How do you score a head? The standard criterion is a Taylor-style importance: estimate the loss increase from masking the head by gradient-times-mask, exactly the Molchanov idea applied to a per-head gate. Practically, you insert a learnable (or constant-1) scalar gate $\xi_j$ multiplying each head's output, run a calibration batch, and score head $j$ by $|\,\partial \mathcal{L}/\partial \xi_j\,|$ — the sensitivity of the loss to turning that head off. Heads with low sensitivity are the redundant ones.

```python
import torch

def head_importance(model, attention_layer, dataloader, num_heads, device):
    """Taylor-style head importance: |d Loss / d head_gate| over a calib set.
    head_gate is a length-num_heads vector of ones multiplying each head out."""
    gate = torch.ones(num_heads, device=device, requires_grad=True)
    attention_layer.head_gate = gate          # multiply head outputs by gate
    scores = torch.zeros(num_heads, device=device)
    model.eval()
    for batch in dataloader:                   # a small calibration set is enough
        model.zero_grad()
        loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
        loss.backward()
        scores += gate.grad.abs().detach()     # accumulate sensitivity
    return scores                              # low score == prunable head
```

Then you remove the lowest-scoring heads — physically, by slicing the Q/K/V/output projection weights — and fine-tune briefly to recover. The `optimum`/`transformers` ecosystem and `torch-pruning`'s attention support both automate the slicing so you get a genuinely smaller model back, not a masked one.

#### Worked example: head-pruning a transformer encoder

Take a BERT-base-sized encoder, 12 layers × 12 heads = 144 heads, fine-tuned on a downstream task. Score every head with the Taylor gate above on a 512-example calibration set. The score distribution is sharply skewed — a handful of heads per layer carry most of the sensitivity, and a long tail is near zero, exactly the Michel et al. finding. Prune the lowest-scoring third (48 of 144 heads, leaving 8 per layer on average), physically slicing the projections, then fine-tune 3 epochs:

- **Heads:** 144 → 96 (−33 percent).
- **Attention-projection FLOPs:** down roughly a third (the projections are now two-thirds the width); attention is a meaningful fraction of a transformer's compute at modest sequence lengths, so end-to-end FLOPs drop by perhaps 12–18 percent depending on sequence length and feed-forward size.
- **Quality:** for a translation model the BLEU might fall from 27.3 to ~26.8 (−0.5); for a classification task accuracy typically falls under a point and often fully recovers after fine-tuning.
- **Latency:** smaller projections mean faster attention; on a CPU target the end-to-end speedup is modest (because the feed-forward layers, untouched here, dominate), maybe 1.1–1.15×. To get a bigger transformer speedup you would *also* prune the feed-forward hidden dimension (a structured neuron prune on the two FFN linears), which is where most transformer FLOPs live — head pruning and FFN-width pruning compose, and together they reach the 1.4–1.6× range.

The honest takeaway: head pruning alone is a smaller lever on transformers than channel pruning is on CNNs, because attention is not where most transformer FLOPs are — the feed-forward blocks are. The biggest structured-pruning wins on transformers come from pruning the FFN hidden width (and, more aggressively, dropping whole layers), with head pruning as a complementary trim. But head pruning is the cleanest illustration of the structured principle in the transformer setting, and it is the right place to start because the redundancy is so well-documented.

### Structured pruning of large language models

The same structured ideas scale up to LLMs, where they matter even more because a 7B model that must run on a laptop or a single consumer GPU has no room to waste. There are three structured dimensions to prune in a decoder-only transformer: **attention heads** (as above), the **FFN intermediate dimension** (the hidden width of the two-matrix feed-forward block, which holds roughly two-thirds of a transformer's parameters and FLOPs), and **whole layers** (depth). LLM-Pruner (Ma et al., 2023) is the reference method: it builds a dependency graph over the LLM (heads, FFN channels, and the coupling between them through the residual stream), scores groups with a Taylor-style importance on a small calibration set, removes the lowest-importance structures, and then recovers with a brief LoRA fine-tune rather than full retraining — because full retraining of an LLM is prohibitively expensive. The recipe is structured pruning's three steps (score, rebuild, fine-tune) with a parameter-efficient recovery step swapped in for the fine-tune.

The honest numbers are sobering and instructive. Structured pruning of LLMs at 20 percent of parameters typically costs a few points of perplexity even after LoRA recovery, and the loss grows fast past 30 percent — the knee is *much* closer to the origin than for an over-parameterized CNN, because large language models are, relative to their training, far less over-parameterized at inference than a vision model trained on a small dataset. This is why the dominant LLM compression lever is *quantization*, not pruning: weight-only int4 (the subject of the [weight-only LLM quantization post on GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)) gives roughly a 4× size cut for under a point of perplexity, a far better trade than structured pruning offers on an LLM. The role of structured pruning on LLMs is therefore as a *complement*: prune a modest amount of depth or width to hit a parameter-count or latency target that quantization alone cannot reach, then quantize the smaller model. Pruning the residual stream width (the model dimension itself) is the most powerful and most dangerous cut, because every layer reads and writes that dimension — it is the LLM equivalent of the residual-stage coupling from section 2, just spanning the entire network. The takeaway: structured pruning works on LLMs, the dependency tracing is the same idea at larger scale, but the knee is early, so use it as a precise trim rather than a primary lever, and always compose it with quantization.

## 9. How far can you go? FLOPs, latency, and accuracy by prune ratio

The single most useful artifact for planning a pruning campaign is the curve of accuracy and latency versus prune ratio, because it shows you the **knee** — the point past which accuracy collapses faster than it is worth. Figure 7 is that curve as a matrix, for the running ResNet-50 example.

![A matrix showing how FLOPs, latency, top-1 accuracy, and recoverability change as channel prune ratio rises from 0 to 30 to 50 to 70 percent, with accuracy holding near baseline until 50 percent and then collapsing at 70 percent](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-7.png)

Reading the matrix top to bottom at each column:

- **0 percent:** baseline. 100 percent FLOPs, 1.00× latency, 76.1 percent top-1.
- **30 percent channels:** FLOPs ≈ 58 percent (quadratic), latency ≈ 0.72×, top-1 ≈ 75.6 percent after fine-tune. This is the comfortable zone — a real speedup for half a point. Fully recoverable.
- **50 percent channels:** FLOPs ≈ 49 percent ($0.5^2$, since both dims halve... actually $(1-0.5)^2 = 0.25$ for the deep interior, but mixed with the linear-only edge layers it lands around 0.49 of total), latency ≈ 0.58×, top-1 ≈ 73.9 percent. You are approaching the knee; recovery is *mostly* there but you are spending 2+ points. Iterative pruning and a longer fine-tune help here.
- **70 percent channels:** FLOPs ≈ 33 percent, latency ≈ 0.41×, top-1 ≈ 68.2 percent. Past the knee. The drop is now severe and only partly recoverable — you have removed channels the network genuinely needed, and no amount of fine-tuning fully covers for them. This is the territory where you should ask whether you want a *pruned* big model or a *purpose-built small* model instead (distillation, or an efficient architecture from the start).

The knee location is architecture- and task-dependent — an over-parameterized net on an easy task has a knee far to the right (you can prune aggressively), while a tight net on a hard task has a knee near the origin (almost any pruning hurts). The practical method is to sweep: prune at several ratios, fine-tune each, plot accuracy vs latency, and pick the Pareto-best point that meets your latency budget. This is the accuracy–efficiency Pareto frontier from the [taxonomy post](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), made concrete for one lever.

A crucial honesty note on the knee: where it sits is partly a statement about how over-parameterized your *starting* model was. A model with a knee at 60 percent was carrying 60 percent dead weight, which tells you the original architecture was the wrong size for the task. Pruning is, in part, a way to *discover* the right size after the fact. If you find yourself routinely pruning 70 percent of a model, the real lesson is that you should have trained a smaller model — and structured pruning is one good way to find out *which* smaller model, by letting importance scores tell you which width the task actually needs.

## 10. Case studies: real numbers from the literature

Four results from the field, to ground the patterns above in published, citable numbers. Where I give a specific figure it is from the cited work; where I round or generalize I say so.

**Li et al. (2017), "Pruning Filters for Efficient ConvNets."** The paper that established L1-norm filter pruning. On VGG-16 / CIFAR-10 they pruned about 34 percent of FLOPs with essentially no accuracy loss after fine-tuning, and on ResNet-110 / CIFAR-10 they cut around 38 percent of FLOPs with negligible loss. The headline contribution was less the exact numbers than the *method*: rank filters by L1 norm, remove the smallest, fine-tune, and crucially, *physically reduce the network* so the FLOP savings are real. This is the template every later filter-pruning paper builds on.

**Liu et al. (2017), "Learning Efficient Convolutional Networks through Network Slimming."** Demonstrated the BatchNorm-γ + L1-penalty recipe across VGG, ResNet, and DenseNet. On VGG-16 / CIFAR-10 they reduced parameters by ~20× and FLOPs by ~50 percent with no accuracy loss, by letting the L1-penalized γ values reveal which channels to drop. The elegance is that channel selection is automatic and learned during a normal-looking training phase, with the only addition being the L1 term on the BN scales. It remains one of the most practical channel-pruning methods precisely because it integrates so cleanly into training.

**He et al. (2019), "Filter Pruning via Geometric Median (FPGM)."** Showed that pruning *redundant* filters (near the geometric median) rather than *small-norm* filters retains accuracy better at aggressive ratios. On ResNet-50 / ImageNet they pruned about 42 percent of FLOPs with roughly a 1-point top-1 drop — a strong point on the Pareto curve at the time — and the gains over norm-based pruning widened as the prune ratio grew, which is exactly where the "all my filters have big norms" failure mode of magnitude pruning bites. FPGM is the criterion to reach for when you are pruning hard.

**Fang et al. (2023), "DepGraph: Towards Any Structural Pruning."** Not a new criterion but the dependency-tracing algorithm that made structured pruning *general*. Before DepGraph, structured pruning of an arbitrary architecture (with its residuals, concats, grouped convs, and reshapes) required a hand-written, architecture-specific script — the source of the silent-corruption bugs from section 2. DepGraph automatically groups coupled parameters across CNNs, transformers, and GNNs, so `torch-pruning` can physically prune almost any model you hand it. This is the engineering unlock that turned structured pruning from "a paper's experiment on VGG" into "a library call on your model," and it is why the code in section 4 is short.

On the transformer side, **Michel et al. (2019), "Are Sixteen Heads Really Better Than One?"** showed that many trained transformers tolerate pruning a large fraction of attention heads — in several layers a single head suffices — with small performance loss, and **Voita et al. (2019)** characterized *which* heads matter (a few specialized ones) versus the prunable tail. Together they are the empirical license for head pruning, and the reason it is the canonical structured-pruning move on transformers.

The common thread across all of these: the win is reported in *FLOPs reduced at a given accuracy*, and the method always *physically shrinks the model*. None of these papers report a sparse-mask result and call it a speedup, because they all know the masked model would not actually run faster on dense hardware. That discipline — report realized FLOPs, rebuild the model, fine-tune — is what separates a structured-pruning result you can ship from an unstructured one you can only put in a slide.

## 11. When to reach for structured pruning (and when not to)

Every technique is a cost, so here is the decisive guidance. Figure 8 is the decision as a tree.

![A decision tree branching on whether you need speed or only file size, then on whether the hardware can skip zeros, leading to structured pruning for dense hardware, 2 to 4 sparsity for sparse Tensor Cores, unstructured for file size only, and a final step to stack quantization on top](/imgs/blogs/structured-pruning-that-actually-speeds-things-up-8.png)

**Reach for structured pruning when:**

- You need *real latency or energy reduction on dense commodity hardware* — a CPU, a phone GPU without sparse support, an embedded board, a microcontroller. This is the common edge case and structured pruning's home turf. It is the *only* pruning flavor that moves the wall clock there.
- Your model is *over-parameterized for the task* — you suspect there is slack to remove. The knee being far to the right is a signal you had too much model, and pruning is a clean way to find the right size.
- You want a model that *stays portable* — a smaller dense model converts to ONNX, TFLite, TensorRT, or Core ML with no special sparse-runtime requirement, carrying its speedup everywhere.

**Do not reach for it when:**

- You only need a *smaller file*, not lower latency. Unstructured pruning reaches far higher sparsity and compresses better; if over-the-air update size is the constraint, it is the better tool.
- Your hardware *can* exploit sparsity. On NVIDIA Ampere+ Tensor Cores, 2:4 structured-sparse mode gives a real speedup at higher sparsity than channel pruning would tolerate — see [N:M sparsity and sparse Tensor Cores](/blog/machine-learning/edge-ai/n-m-sparsity-and-sparse-tensor-cores). On that silicon, the constrained-sparse path can beat channel pruning.
- You need *very* high compression and can afford to retrain from scratch. Past the knee, a *purpose-built small architecture* (or a distilled student) often beats an aggressively pruned big model at the same size, because it was designed to be small rather than carved down to small.
- The accuracy budget is razor-thin and you have not exhausted cheaper levers. Always run a good compiler/runtime first (free speedup, zero accuracy cost), and consider quantization, which often gives a larger speedup per accuracy point than pruning on int8-capable hardware.

The strongest move is usually to **compose** structured pruning with quantization. They pull different levers — pruning removes channels (fewer, smaller dense tensors), quantization reduces bits per number — and they stack nearly multiplicatively because they shrink different things. The canonical recipe: structured-prune to hit your FLOP/latency target, fine-tune to recover accuracy, *then* quantize the smaller model to int8 (PTQ or QAT) for a further size and speed cut. Quantize last, because quantization is sensitive to the weight distribution and you want to quantize the *final* distribution after pruning has settled it. The [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) lays out the full composition order, and the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) walks an end-to-end deployment that stacks these levers in sequence on a single model.

### A stress test of the decision

Let us pressure-test the recommendation against the hard cases. *What if the pruned model is memory-bound, not compute-bound?* Structured pruning still helps, because it shrinks the activation tensors and thus the bytes moved — unlike unstructured, which moves the same bytes. *What if you prune so thin that the layers underutilize the SIMD width?* Then realized speedup falls below the FLOP cut (very thin convs waste wide vector units), which is a real ceiling — the fix is to prune less aggressively or to prune in hardware-friendly multiples (e.g. channel counts that stay a multiple of 8 or 16 so the kernels stay efficient). *What if a downstream op does not support the new shape on the NPU and it falls back to CPU?* That is the portability trap — a smaller dense model is *more* likely to stay on the fast path than a sparse one, but you should always re-profile on the actual target after pruning, because a fused kernel that was fast at the original width may not have a tuned variant at your pruned width. *What if fine-tuning does not recover the accuracy?* You are past the knee for this architecture and task — back off the ratio, go iterative, or switch to distilling a purpose-built small model. The decision tree's branches are not academic; each one is a real fork I have hit in production.

## 12. Key takeaways

- **Unstructured pruning is a storage win, not a speed win, on dense hardware.** Scattered zeros are still multiplied at full price by stock dense kernels. If you measured no speedup after pruning, this is almost certainly why.
- **Structured pruning removes whole units — channels, neurons, heads, blocks — so the dense tensors actually shrink.** The result is a smaller dense model that is genuinely faster on every backend, with no special sparse kernel.
- **Channel pruning is quadratic in the prune ratio.** Pruning a fraction $p$ of channels everywhere cuts interior-layer FLOPs by roughly $(1-p)^2$, because both the output channels of a layer and the input channels of the next shrink together. A 30 percent channel cut is closer to a 50 percent FLOP cut.
- **The dependency problem is the hard part.** Removing a channel forces matching cuts in BatchNorm, the next layer's input, and any residual or concat coupled to it. Trace the dependency graph (use `torch-pruning`'s DepGraph) and prune coupled groups together, or you get silent corruption.
- **Mask for research, rebuild for speed.** PyTorch's `prune` API masks (same shape, same speed); to get latency you must physically rebuild the model with smaller tensors. Always verify the tensor shape actually changed.
- **Criterion matters less than you fear.** L2 norm is a free, strong baseline; Taylor importance retains accuracy best at high sparsity; Network Slimming gives learned importance via BatchNorm γ; FPGM removes redundancy norm-methods miss. Start cheap, escalate only if you need to push past the knee.
- **Prune, rebuild, fine-tune, repeat.** Accuracy drops at prune time and fine-tuning recovers most of it; go iterative for aggressive sparsity. Never report a pruned model's accuracy without the fine-tune.
- **Measure on the real target, batch 1, warm, thermally settled.** FLOPs set the ceiling; the device sets the truth. Realized speedup is sublinear in the FLOP cut.
- **Compose with quantization, prune before you quantize.** The levers stack nearly multiplicatively; quantize the final pruned distribution last.

## Further reading

- **Li, Kadav, Durdanovic, Samet, Graf (2017), "Pruning Filters for Efficient ConvNets"** — the L1-norm filter-pruning template that physically reduces the network.
- **Liu, Li, Shen, Huang, Yan, Zhang (2017), "Learning Efficient Convolutional Networks through Network Slimming"** — channel pruning via an L1 penalty on BatchNorm γ; learned, automatic channel selection.
- **Molchanov, Tyree, Karras, Aila, Kautz (2017) and Molchanov, Mallya, Tyree, Frosio, Kautz (2019), "Importance Estimation for Neural Network Pruning"** — Taylor-expansion importance, the most principled criterion.
- **He, Liu, Wang, Hu, Yang (2019), "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration"** — FPGM, pruning redundancy rather than small magnitude.
- **Fang, Ma, Song, Mi, Wang (2023), "DepGraph: Towards Any Structural Pruning"** — the dependency-tracing algorithm behind `torch-pruning`; structured pruning for arbitrary architectures.
- **Michel, Levy, Neubig (2019), "Are Sixteen Heads Really Better Than One?"** and **Voita, Talbot, Moiseev, Sennrich, Titov (2019), "Analyzing Multi-Head Self-Attention"** — the empirical case for attention-head pruning.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame and composition order; [unstructured pruning and the lottery ticket](/blog/machine-learning/edge-ai/unstructured-pruning-and-the-lottery-ticket) for sparsity as a phenomenon; [N:M sparsity and sparse Tensor Cores](/blog/machine-learning/edge-ai/n-m-sparsity-and-sparse-tensor-cores) for the hardware that *can* exploit sparsity; [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for why FLOPs are not latency; and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for stacking pruning with the other levers end to end.
