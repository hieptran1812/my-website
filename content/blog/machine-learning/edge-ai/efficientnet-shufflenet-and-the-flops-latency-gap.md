---
title: "EfficientNet, ShuffleNet, and why low FLOPs don't mean low latency"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn why two models with identical FLOPs can run twice as far apart on the same chip, how EfficientNet's compound scaling and ShuffleNet's shuffle trick really work, and how to optimize for measured latency instead of the metric that lies."
tags:
  [
    "edge-ai",
    "model-optimization",
    "efficientnet",
    "shufflenet",
    "efficient-architecture",
    "flops",
    "latency",
    "inference",
    "efficient-ml",
    "neural-architecture-search",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-1.png"
---

I once shipped a model that, on paper, should have been twice as fast as the one it replaced. The replacement had almost exactly half the FLOPs. The slide deck was beautiful: a clean Pareto plot, our new architecture sitting comfortably below and to the right of the old one, fewer multiply-adds for the same accuracy. Everyone nodded. We swapped it in. The p50 latency on the actual phone went *up* by eleven percent.

That sting — the gap between the FLOP count that wins the architecture review and the wall-clock latency the user actually feels — is the single most expensive misunderstanding in efficient deep learning. It is responsible for more wasted research months than any other mistake I have watched teams make. Two models with identical FLOPs can differ by 2× in latency on the very same chip. A "more efficient" block can be slower than the dense one it replaced. And the FLOP-optimal architecture from a NAS run can lose, badly, to a chunkier design that happens to keep the silicon fed.

By the end of this post you will understand exactly why this happens and what to do about it. We will derive **EfficientNet's compound scaling** — the surprisingly clean idea that you should grow a network's depth, width, and input resolution *together* under a fixed budget, and the constraint $\alpha\cdot\beta^2\cdot\gamma^2\approx 2$ that makes it work. We will take apart **ShuffleNet's** grouped-convolution-plus-channel-shuffle trick, and its quieter cousin **GhostNet**, which manufactures feature maps on the cheap. And then we will get to the heart of it: the **four ShuffleNetV2 guidelines** (Ma et al., 2018), which explain — with measured numbers, not hand-waving — *why FLOPs lie about latency*. Memory access cost, not arithmetic, is what kills you on a phone. Figure 1 shows the whole problem in one frame: two blocks the FLOP counter swears are identical, running 2× apart on one chip.

![A before and after comparison showing a dense convolution block and a fragmented depthwise block with identical 300 million FLOPs but the second running roughly twice as slow on the same Pixel CPU](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-1.png)

This is the **efficient-architecture** lever from our [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — the one that changes the *operations themselves* rather than the bits (quantization) or the structure (pruning) or the topology via a teacher (distillation). It is also the lever most poisoned by bad metrics, because architectures are designed and compared offline, long before anyone measures them on the target. The cure is the discipline this whole series keeps coming back to: optimize the **direct metric on the target device**, not a convenient proxy. Two prerequisites worth reading first if you have not: [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) and [the roofline model where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), because the punchline of this post is a roofline argument wearing an architecture costume.

## 1. The crime scene: a FLOP count that lied to me

Let me reconstruct the failure from the intro precisely, because the anatomy generalizes. The old block was a standard `3×3` convolution with 256 input channels and 256 output channels on a `28×28` feature map. The new block was a depthwise-separable replacement: a `3×3` depthwise conv followed by a `1×1` pointwise conv, the MobileNet recipe. On a FLOP basis the separable block is dramatically cheaper — roughly an 8-9× reduction for this shape. But I had not replaced one dense block with one separable block. I had replaced one dense block with a *stack* of cheap blocks, tuned so the FLOPs came out to half the original. And the stack was full of small, fragmented operations: many parallel branches, several depthwise convs, a pile of element-wise additions and channel concatenations stitching it all together.

Here is what the FLOP counter could not see. Each of those depthwise convs does almost no arithmetic per byte it touches. A depthwise `3×3` does 9 multiply-adds per output element, full stop — it does not mix channels, so there is no inner-product-over-channels loop to amortize the memory traffic against. The block read activations from DRAM, did a trickle of math, and wrote them back, over and over. The chip spent most of its time *waiting for memory*, not computing. The old dense conv, by contrast, did a fat matrix multiply per output pixel: 256 multiply-adds accumulated per output channel, reusing each loaded input value across all 256 output channels. High arithmetic intensity. The tensor units stayed busy. It was *compute-bound* — and on this chip, compute was fast and memory was slow.

So the "8× cheaper" block was cheaper in the currency the FLOP counter measures (arithmetic) and *more expensive* in the currency the chip actually charges (memory traffic). That is the entire post in one sentence, and everything below is a careful unpacking of why it is true and what the field did about it.

To be precise about the vocabulary, since the whole argument hinges on it:

- **FLOPs** (or more honestly **MACs**, multiply-accumulate operations — one MAC is two FLOPs, an add and a multiply, but the literature sloppily says "FLOPs" for both) count the *arithmetic*. They are a property of the math the model expresses, independent of hardware.
- **MAC** in the ShuffleNetV2 sense is an unfortunate name collision: there, **MAC = memory access cost**, the number of bytes the operation must read and write. This counts the *traffic*. To avoid the collision I will always write "memory access cost" in full when I mean the ShuffleNetV2 quantity, and "MACs" only for multiply-accumulates.
- **Arithmetic intensity** is the ratio: FLOPs per byte of memory traffic. It is the single number that decides whether an operation is compute-bound (high intensity, limited by how fast the chip multiplies) or memory-bound (low intensity, limited by how fast the chip can move bytes). The roofline model, which we lean on hard later, is just a plot of achievable throughput against arithmetic intensity.

Hold those three apart and the rest is mechanical. The FLOP count is the numerator of arithmetic intensity. Latency, on memory-bound ops, is governed by the denominator. Optimizing the numerator while ignoring the denominator is how you ship a slower model with a better slide.

## 2. EfficientNet: scale all three dimensions, together

Before we get to why FLOPs lie, let me give you the other half of the story — the half that is genuinely beautiful, and that the field got *right*. EfficientNet (Tan & Le, ICML 2019) asked a simple question that nobody had answered cleanly: when you have more compute budget and want a more accurate model, *what do you spend it on?* You can make the network **deeper** (more layers), **wider** (more channels per layer), or feed it **higher-resolution** images. Pre-EfficientNet, people scaled one of these at a time, by feel. ResNet went from ResNet-18 to ResNet-200 by stacking depth. Wide ResNet went wider. GPipe scaled resolution. Each worked a little, then saturated.

The EfficientNet insight is that these three dimensions are not independent, and scaling them in balance is strictly better than scaling any one of them alone. Figure 2 lays out the three dimensions and what each one buys.

![A matrix figure showing the three scaling dimensions depth width and resolution, how each grows as a base raised to the coefficient phi, what each adds to the model, and the specific capacity bottleneck each one lifts](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-2.png)

The intuition for *why* balance wins is worth sitting with. Suppose you only scale depth. More layers give the network a larger receptive field and more compositional power — but if the input is still `224×224` and the channels are still narrow, the deep layers run out of fine-grained features to compose and fresh spatial detail to attend to. Depth saturates because the *other two dimensions are starving it*. Symmetrically, scale only resolution and you feed the network a richer image, but a shallow, narrow network cannot extract or hold the extra detail; it down-samples it away in the first few blocks. Scale only width and you get more feature channels per layer, but without depth they stay shallow, low-order features. Each dimension hits a ceiling set by the others. The way out is to lift all three ceilings at once.

### 2.1 The compound scaling rule

EfficientNet formalizes this with a single **compound coefficient** $\phi$ that controls how much extra budget you are spending, and three base coefficients $\alpha,\beta,\gamma$ that say how to split that budget across the three dimensions:

$$
\text{depth: } d = \alpha^{\phi}, \qquad
\text{width: } w = \beta^{\phi}, \qquad
\text{resolution: } r = \gamma^{\phi}.
$$

The bases satisfy the constraint

$$
\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2, \qquad \alpha \ge 1,\ \beta \ge 1,\ \gamma \ge 1.
$$

You find $\alpha,\beta,\gamma$ once, by a small grid search on the baseline (EfficientNet uses $\alpha=1.2,\ \beta=1.1,\ \gamma=1.15$). Then you scale the whole family by just turning the dial $\phi$. Set $\phi=0$ and you have the baseline B0; set $\phi=1,2,\dots$ and you walk up to B1, B2, all the way to B7. One knob, the entire family.

### 2.2 Why the constraint is $\alpha\cdot\beta^2\cdot\gamma^2\approx 2$, derived

This is the part most explanations skip, and it is the most satisfying part. The constant 2 and the squares are not arbitrary — they fall directly out of how FLOPs scale with each dimension.

Start by counting FLOPs for a convolutional network as a function of the three scaling factors, relative to the baseline:

- **Depth** $d$: doubling the number of layers doubles the FLOPs (twice as many convolutions to run). So FLOPs scale **linearly** in $d$. Factor: $d$.
- **Width** $w$: a convolution's cost is proportional to (input channels) × (output channels). If you scale both input and output channels by $w$, you multiply the per-layer cost by $w \times w = w^2$. So FLOPs scale **quadratically** in width. Factor: $w^2$.
- **Resolution** $r$: a convolution runs once per output spatial location. If you scale both height and width of the feature maps by $r$, you have $r \times r = r^2$ as many locations. So FLOPs scale **quadratically** in resolution. Factor: $r^2$.

Multiply them together and the total FLOPs of a scaled network, relative to the baseline, is

$$
\text{FLOPs} \ \propto\ d \cdot w^{2} \cdot r^{2}.
$$

Now substitute the compound-scaling parameterization $d=\alpha^\phi,\ w=\beta^\phi,\ r=\gamma^\phi$:

$$
\text{FLOPs} \ \propto\ \alpha^{\phi} \cdot \big(\beta^{\phi}\big)^{2} \cdot \big(\gamma^{\phi}\big)^{2}
= \big(\alpha \cdot \beta^{2} \cdot \gamma^{2}\big)^{\phi}.
$$

There it is. If you *constrain* the bracketed base term to equal 2, then

$$
\text{FLOPs} \ \propto\ 2^{\phi}.
$$

The total compute *doubles every time you increment $\phi$ by one.* That is exactly the property you want from a scaling dial: each step costs a predictable, constant *factor* more, and the constant is a clean 2. If the constraint product were 4, FLOPs would quadruple per step; if it were 1.5, they would grow by 1.5× per step. The choice of 2 is just a convention that makes "one step of $\phi$" mean "twice the FLOPs," so the family spans a wide compute range in a handful of steps. The squares on $\beta$ and $\gamma$ are not a choice at all — they are forced by the quadratic FLOP dependence on width and resolution. The geometry of convolution wrote the constraint.

This is also why you cannot just crank $\gamma$ (resolution) to the moon even though higher resolution helps accuracy: resolution costs $r^2$, so it eats the budget fastest. The grid-searched $\gamma=1.15$ (versus $\alpha=1.2$ for depth) reflects exactly this — resolution is given a *smaller* base because each unit of it is more expensive in FLOPs, and the constraint keeps the product balanced.

### 2.3 Compound vs single-dimension scaling, at matched FLOPs

The constraint guarantees both strategies spend the same FLOPs per $\phi$ step. The question is which spends them better. Figure 3 contrasts dumping the whole budget into depth against the balanced split.

![A before and after figure contrasting scaling depth alone, which saturates accuracy, against balanced compound scaling under the constraint, which reaches a higher accuracy ceiling at the same FLOP budget](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-3.png)

The empirical result from the EfficientNet paper is decisive and worth memorizing. Starting from the same baseline and spending the same extra FLOPs, compound scaling delivers roughly **2 to 3 points more ImageNet top-1 accuracy** than scaling any single dimension. In the paper's controlled experiment on a fixed baseline at matched FLOPs, scaling width alone, depth alone, or resolution alone each plateaus around 80% top-1, while compound scaling pushes past it. The reason is the ceiling argument from above: single-dimension scaling hits diminishing returns because the other two dimensions starve the one you are growing; balanced scaling lifts all the ceilings together, so the returns keep coming.

#### Worked example: spending a 4× FLOP budget two ways

Say you start from EfficientNet-B0 (0.39B FLOPs) and you are handed a 4× FLOP budget — about 1.6B FLOPs — and asked to produce the most accurate model you can.

Option A, depth only. FLOPs scale linearly in depth, so 4× FLOPs means 4× the layers. B0 has 18 mobile inverted bottleneck (MBConv) blocks across its stages; this option gives you about 72. You keep `224×224` input and the original channel widths. In practice this lands around 80% top-1 — the deep stack composes the same narrow, low-resolution features it always had, and the marginal layers add little.

Option B, compound. Solve $2^\phi = 4 \Rightarrow \phi = 2$. Then depth $= 1.2^{2} = 1.44\times$, width $= 1.1^{2} = 1.21\times$, resolution $= 1.15^{2} = 1.32\times$. Check the FLOP budget: $1.44 \times 1.21^2 \times 1.32^2 = 1.44 \times 1.46 \times 1.74 \approx 3.66\times \approx 4\times$ (the constraint is approximate, so it lands close to but not exactly 4×). This is essentially EfficientNet-B2, which the paper reports at about **80.1% top-1** versus B0's 77.1 — and it does it with 9.1M params and 1.0B FLOPs, beating much heavier hand-designed nets. Same budget, balanced split, several points higher. The input grew to `260×260`, the channels widened modestly, the network deepened modestly — every dimension lifted, none starved.

That is compound scaling. It is the right way to *grow* a model when FLOPs are an acceptable proxy for cost — which, crucially, they are on a fat cloud GPU. The trouble starts when you take a FLOP-optimal design to a phone.

### 2.4 Where B0 comes from, and what it is made of

One question the formulas above leave open: scaling *what*? Compound scaling is a multiplier, and a multiplier needs a baseline to multiply. EfficientNet-B0, the baseline, was itself produced by a neural architecture search — the same multi-objective search that produced MnasNet, optimizing for accuracy and a *latency-like* FLOP target jointly. So the EfficientNet recipe is really two stages: search for a good small base (B0), then scale it up compoundly (B1-B7). The compound-scaling math is the elegant, transferable part; the B0 base is the un-glamorous engineering that makes the scaling start from a strong point. If you scale a mediocre baseline compoundly you get a mediocre family — the scaling preserves the baseline's efficiency, it does not create it.

B0's building block is the **MBConv** — the mobile inverted bottleneck convolution, borrowed from MobileNetV2 and augmented with squeeze-and-excite. It does four things in sequence: an *expansion* `1×1` conv that widens the channel count by a factor (typically 6×), a depthwise `3×3` (or `5×5`) conv that filters spatially in the expanded space, a squeeze-and-excite module that reweights channels with a tiny global attention, and a *projection* `1×1` conv that squeezes back down to the bottleneck width, with a residual add when the shapes match. The "inverted" part is that, unlike a classic ResNet bottleneck which narrows then widens, MBConv widens then narrows — the expensive depthwise conv runs in the *wide* space where there is more to filter, and the cheap `1×1`s do the widening and narrowing.

Hold onto the squeeze-and-excite and the residual add, because they matter for the latency story. Squeeze-and-excite is a global-average-pool followed by two small fully-connected layers and a per-channel multiply — almost no FLOPs, but it is an element-wise rescale of the whole feature map (a G4 offender) and it forces a global reduction that serializes the pipeline. The swish activation EfficientNet uses ($x\cdot\sigma(x)$) is similarly cheap in FLOPs and expensive in element-wise memory traffic, and it quantizes worse than ReLU. These are exactly the choices EfficientNet-Lite later stripped out for mobile. So B0 is a network whose every block contains two structural traps the FLOP count cannot see — which is why B0 ports to the edge less cleanly than its lovely FLOP number suggests, a thread we pick up in the case studies.

## 3. The efficient-block zoo: depthwise-separable, grouped, ghost

EfficientNet scales a network; the next three ideas redesign the *block* itself to do more with fewer multiply-adds. To see why FLOPs mislead, you first have to understand what these blocks actually do, because each one trades dense arithmetic for something the FLOP count loves and the memory system may hate. The whole family of these tricks is the subject of a dedicated post in this series, [building blocks for efficient models](/blog/machine-learning/edge-ai/building-blocks-for-efficient-models); here I cover just enough to make the latency argument land.

### 3.1 Depthwise-separable convolution (the MobileNet idea)

A standard convolution with kernel `k×k`, $C_{in}$ input channels, and $C_{out}$ output channels, applied to an `H×W` feature map, costs

$$
\text{FLOPs}_{\text{std}} = H \cdot W \cdot C_{in} \cdot C_{out} \cdot k^2
$$

multiply-accumulates. It does two jobs at once: it filters spatially (the `k×k` window) and it mixes channels (the sum over $C_{in}$). Depthwise-separable convolution factors these apart. A **depthwise** conv applies one `k×k` filter per input channel — spatial filtering, no channel mixing — at cost $H\cdot W\cdot C_{in}\cdot k^2$. Then a **pointwise** `1×1` conv mixes channels at cost $H\cdot W\cdot C_{in}\cdot C_{out}$. The total:

$$
\text{FLOPs}_{\text{sep}} = H\cdot W\cdot C_{in}\cdot k^2 \;+\; H\cdot W\cdot C_{in}\cdot C_{out}.
$$

The ratio of separable to standard is

$$
\frac{\text{FLOPs}_{\text{sep}}}{\text{FLOPs}_{\text{std}}} = \frac{1}{C_{out}} + \frac{1}{k^2}.
$$

For a `3×3` kernel ($k^2=9$) and any reasonable channel count, this is roughly $1/9$ — about an 8 to 9× FLOP reduction. The whole [MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) is built on this factorization, and EfficientNet's MBConv block is a depthwise-separable inverted bottleneck. On a FLOP basis it is a slam dunk. Remember the term $1/k^2$ though: the pointwise `1×1` does most of the *remaining* work, and as we will see, that `1×1` is where the memory access cost lives.

### 3.2 Grouped convolution and ShuffleNet's shuffle

Grouped convolution goes further on the channel-mixing cost. Instead of every output channel seeing every input channel, you split the channels into $g$ groups and convolve within each group independently. This cuts the channel-mixing FLOPs by a factor of $g$: a grouped `1×1` conv costs $H\cdot W\cdot C_{in}\cdot C_{out}/g$. The catch is obvious once you say it out loud — if groups never talk, the network fragments into $g$ independent sub-networks and information cannot flow across groups. Stack grouped convs and a feature computed in group 1 can never influence group 2's output. Accuracy collapses.

ShuffleNet V1 (Zhang et al., CVPR 2018) fixes this with a parameter-free, FLOP-free trick: the **channel shuffle**. After a grouped conv produces $g$ groups of channels, you *permute* the channels so that the next grouped conv's groups each draw from all of the previous groups. Concretely, if you have $g$ groups of $n$ channels each, you reshape the channel dimension to $(g, n)$, transpose it to $(n, g)$, and flatten — a deterministic permutation that interleaves the groups. The next grouped conv now sees a mixture, so cross-group information flows even though no single conv ever mixes all channels. Figure 4 shows the dataflow: grouped convolutions trap features inside their group, and the shuffle permutes the outputs so the following layer sees a blend.

![A graph figure showing input channels split into groups, two grouped convolutions whose outputs are trapped per group, then a channel shuffle that permutes and regroups so the next grouped convolution sees a mixture of all groups](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-4.png)

The shuffle costs *zero FLOPs*. It is pure data movement — a permutation of the channel axis. And this is your first real foreshadowing of the whole problem: an operation with zero FLOPs is not free. The shuffle reads the entire activation tensor and writes it back in a different order. On a bandwidth-bound chip that read-modify-write of the full tensor can cost as much wall-clock time as a small convolution. The FLOP count says "free." The memory system says "pay up."

### 3.3 GhostNet: feature maps on the cheap

GhostNet (Han et al., CVPR 2020) attacks redundancy in feature maps directly. The observation is that many channels in a trained CNN's feature maps are near-duplicates of each other — cheap linear transforms of a few "intrinsic" maps. So why compute all of them with expensive convolutions? GhostNet computes a small set of intrinsic feature maps with a normal (but reduced) convolution, then generates the rest — the "ghost" maps — by applying cheap *linear operations* (typically small depthwise convolutions) to the intrinsic ones. If you want $C_{out}$ output channels, you produce $C_{out}/s$ intrinsic maps with a real conv and the remaining $(s-1)/s$ fraction as ghosts, for a roughly $s\times$ FLOP and parameter reduction.

GhostNet is a lovely idea and it does deliver real speedups on some hardware. But it is also a cautionary tale for this post: the cheap linear ops are, again, depthwise-style operations with low arithmetic intensity, and the concatenation of intrinsic and ghost maps is element-wise data movement. Whether GhostNet is actually faster than a dense baseline depends entirely on whether the target chip is bandwidth-bound — exactly the question FLOPs cannot answer. On a GPU with abundant compute and tight memory it can underperform its FLOP count; on a CPU it tends to do better. Same model, different verdict, depending on the silicon. We are now ready to make that dependence rigorous.

### 3.4 Four blocks, four trade-offs

Before the rigor, a side-by-side of the blocks we have met, because the pattern across them is the whole point. Every one of these *reduces FLOPs by trading dense arithmetic for something cheaper-in-FLOPs but data-movement-heavy* — and every one therefore has a hidden latency liability that depends on the chip.

| Block | FLOP-saving trick | What it costs in latency terms | When it actually wins |
| --- | --- | --- | --- |
| Standard `3×3` conv | none (baseline) | high FLOPs, but high arithmetic intensity | compute-bound chips; the speed-per-FLOP champion |
| Depthwise-separable | factor spatial filtering from channel mixing | depthwise is memory-bound; pointwise `1×1` carries the traffic | when you are FLOP-limited, not bandwidth-limited |
| Grouped conv + shuffle | divide channel mixing across `g` groups | shuffle moves the whole tensor at zero FLOPs (G4); extra groups inflate traffic (G2) | moderate `g` on CPU; rarely high `g` on GPU |
| Ghost module | generate redundant maps by cheap linear ops | the cheap ops are low-intensity; concat is element-wise | CPU and bandwidth-rich chips; underperforms on compute-tight GPUs |

Read the third column top to bottom: *memory-bound, memory-bound, memory-bound.* Every FLOP-reduction trick in the modern efficient-CV toolkit pays for its cheaper arithmetic in data movement, and data movement is the currency the FLOP count refuses to bill. That is not a coincidence or a flaw in any one design — it is the structural reason the entire category of "low-FLOP" blocks behaves erratically across hardware. With that pattern named, the four guidelines stop being a list to memorize and become predictions you could have made yourself.

## 4. Why FLOPs lie: memory access cost and the four guidelines

Here is the central science of the post. In 2018, the ShuffleNet authors (Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," ECCV 2018) did something the field should have done years earlier: they stopped trusting FLOPs and *measured*. They built dozens of block variants with matched FLOPs and timed them on real hardware — an ARM CPU (a phone) and a GPU. The variance was enormous. Blocks the FLOP counter called identical ran up to 2× apart. From the measurements they extracted four guidelines, each one a place where FLOPs and latency diverge. Figure 5 is the whole framework as a table; we will walk it row by row.

![A matrix figure listing the four ShuffleNetV2 guidelines, what each one warns against, why the FLOP count fails to capture it, and the concrete fix that keeps on-device latency low](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-5.png)

### 4.1 The model behind it: latency = compute + memory + overhead

Start with a decomposition. The wall-clock time of an operation is, to first order,

$$
T \approx \max\!\big(T_{\text{compute}},\ T_{\text{memory}}\big) + T_{\text{overhead}},
$$

where $T_{\text{compute}} = \text{FLOPs} / \text{(peak FLOP/s)}$, $T_{\text{memory}} = \text{(bytes moved)} / \text{(bandwidth)}$, and $T_{\text{overhead}}$ is kernel-launch and scheduling cost. The $\max$ is the roofline: an op is bound by whichever of compute or memory is the bottleneck. FLOPs only appear in $T_{\text{compute}}$. If your op is memory-bound — if $T_{\text{memory}} > T_{\text{compute}}$ — then *FLOPs do not appear in the latency at all.* You can halve the FLOPs and the latency will not budge, because the chip is sitting idle waiting for DRAM either way. That single inequality is why FLOPs lie. The four guidelines are four specific ways real blocks fall on the wrong side of it.

### 4.2 G1 — equal channel width minimizes memory access cost

Consider the workhorse of every efficient net: the `1×1` (pointwise) convolution. With $C_{in}$ input channels and $C_{out}$ output channels on an `H×W` map, its FLOPs are $B = H\cdot W\cdot C_{in}\cdot C_{out}$ (call it $B$). Its memory access cost — bytes read plus written, counting input activations, output activations, and the `1×1` weights — is, in elements,

$$
\text{MAC} = \underbrace{H\cdot W\cdot C_{in}}_{\text{read input}} + \underbrace{H\cdot W\cdot C_{out}}_{\text{write output}} + \underbrace{C_{in}\cdot C_{out}}_{\text{weights}}.
$$

Now fix the FLOPs $B = H W C_{in} C_{out}$ and ask: which split of channels into $C_{in}, C_{out}$ minimizes the memory access cost? Let $hw = H\cdot W$. By the arithmetic-mean–geometric-mean inequality,

$$
\text{MAC} = hw\,(C_{in} + C_{out}) + C_{in}C_{out} \ \ge\ hw\cdot 2\sqrt{C_{in}C_{out}} + C_{in}C_{out}.
$$

Since $C_{in}C_{out} = B/hw$ is fixed by the FLOP constraint, $\sqrt{C_{in}C_{out}}$ is fixed too, and the bound is *achieved with equality exactly when $C_{in} = C_{out}$.* In words: for a fixed FLOP budget and feature-map size, a `1×1` convolution touches the least memory when its input and output channel counts are equal. Any imbalance — the wide-then-narrow squeezes that bottleneck blocks love — moves more bytes for the same arithmetic, and on a memory-bound chip that means more latency.

The ShuffleNetV2 measurements confirm it crisply: a stack of `1×1` convs with a 1:1 channel ratio ran meaningfully faster than the same FLOPs split 1:2 or 2:1, and the gap widened as the imbalance grew. The lesson designed into ShuffleNetV2's block is to keep the two `1×1` convs at equal width. FLOPs are blind to this entirely — they depend only on the *product* $C_{in}C_{out}$, not the split.

#### Worked example: the same `1×1` conv, balanced vs lopsided

Take $H=W=14$, so $hw = 196$, and a FLOP budget fixing $C_{in}C_{out} = 256^2 = 65{,}536$.

- Balanced, $C_{in}=C_{out}=256$: $\text{MAC} = 196\cdot(256+256) + 65{,}536 = 196\cdot 512 + 65{,}536 = 100{,}352 + 65{,}536 = 165{,}888$ elements.
- Lopsided, $C_{in}=512,\ C_{out}=128$: same product $512\cdot128 = 65{,}536$, same FLOPs. $\text{MAC} = 196\cdot(512+128) + 65{,}536 = 196\cdot 640 + 65{,}536 = 125{,}440 + 65{,}536 = 190{,}976$ elements.

The lopsided block moves about **15% more bytes for identical FLOPs**. On a memory-bound `1×1` conv, that is roughly 15% more latency, invisible to any FLOP-based comparison. Push the imbalance to 1024:64 and the gap grows further. The FLOP counter rates these blocks as exactly tied.

### 4.3 G2 — excessive grouped convolution raises memory access cost

Grouped convolution is the FLOP-reduction hero of section 3, and G2 is its bill. A grouped `1×1` conv with $g$ groups has FLOPs $B = H W C_{in} C_{out} / g$ — the group count divides the arithmetic. But its memory access cost does *not* shrink proportionally. The activations still have to be fully read and written ($H W C_{in}$ in, $H W C_{out}$ out, unchanged by $g$); only the weight term drops to $C_{in}C_{out}/g$. So for a *fixed FLOP budget* $B$, increasing $g$ lets you afford more channels (since FLOPs $= HWC_{in}C_{out}/g$, holding $B$ fixed means $C_{in}C_{out}$ grows linearly with $g$), which *inflates* the activation memory traffic. Working it through, the memory access cost for fixed FLOPs $B$ and fixed input grows roughly as

$$
\text{MAC} \approx hw\,C_{in} + \frac{B\,g}{hw\,C_{in}} \cdot hw \;+\; \frac{B}{C_{in}} = hw\,C_{in} + \frac{B\,g}{C_{in}} + \frac{B}{C_{in}},
$$

so the dominant variable term grows *linearly in $g$*. More groups, more bytes, for the same FLOPs. The ShuffleNetV2 measurements show a grouped `1×1` with $g=8$ running substantially slower than $g=1$ at matched FLOPs on the GPU — the extra channels that the FLOP saving "bought" cost real bandwidth. The guideline: use grouped convolution, but do not crank $g$ to extremes chasing FLOP savings; the latency you spend on memory traffic will erase the compute you saved. ShuffleNetV2 actually steps *back* from V1's aggressive grouping for exactly this reason.

### 4.4 G3 — network fragmentation hurts parallelism

This is the one that bit me in the intro. "Fragmentation" means splitting an operation into many small ones — many parallel branches, many tiny convs, the multi-path blocks that Inception and NASNet-style architectures love and that FLOP-minimizing NAS tends to discover, because lots of small cheap ops can hit a low FLOP target. Each small op, though, carries fixed overhead: a kernel launch, scheduling, synchronization at the merge. And small ops cannot saturate a wide parallel processor — a GPU with thousands of cores or an NPU with a big systolic array runs a tiny conv at a fraction of peak, then waits.

The ShuffleNetV2 paper measured this directly: a block with 4 fragmented branches ran roughly **3× slower on a GPU** than a single equivalent block with the same total FLOPs, and noticeably slower on the CPU too. The FLOP count is identical — it sums the arithmetic regardless of how it is chopped up. But the device sees $T_{\text{overhead}}$ multiply by the number of fragments and the parallel efficiency collapse. The guideline: prefer fewer, larger operations over many small ones. A chunky block that keeps the silicon's parallel units full beats an elegant fan-out of cheap branches, even at equal or *higher* FLOPs. This is the deepest reason hand-designed FLOP-efficient blocks so often disappoint on hardware, and why latency-aware NAS produces less fragmented, "boring," but faster networks.

### 4.5 G4 — element-wise operations are not free

ReLU, the residual add, tensor copies, the channel shuffle, bias adds, depthwise convs (which are element-wise-ish in their memory profile) — these have negligible FLOPs but real memory access cost. An element-wise add of two tensors reads two full tensors and writes one: three tensor-sized memory transactions for zero useful arithmetic by FLOP accounting. The ShuffleNetV2 experiment removed the ReLU and the shortcut-add from a bottleneck block and measured about a **20% speedup on both CPU and GPU** — a fifth of the runtime was being spent on operations the FLOP count valued at essentially zero.

This is why modern efficient blocks fuse aggressively (folding ReLU and bias into the preceding conv so the activation never round-trips through DRAM) and why ShuffleNetV2's block uses a channel *split and concat* instead of an *add* for its residual path — concat avoids the read-modify-write that the additive shortcut forces. Figure 6 makes the FLOPs-versus-traffic split concrete on the canonical offender, the depthwise conv: tiny arithmetic, full-tensor read and write, bandwidth-bound.

#### Worked example: two equal-FLOP blocks, measured apart

Make the gap concrete with a representative measurement on an ARM mobile CPU at batch=1. Take two blocks designed to the *same* FLOP target of about 140 MFLOP on a `28×28×144` feature map.

- **Block A — single fused path.** One depthwise `3×3` plus one `1×1` projection, output written once, ReLU fused into the conv. No residual add, no shuffle, no branches. Three kernels collapse to two after fusion. Measured p50: about **3.0 ms**.
- **Block B — the same FLOPs, "elegantly" decomposed.** Two parallel branches each doing half the channels, a channel shuffle to mix them, a separate ReLU pass, and an additive residual shortcut. Same 140 MFLOP by the counter. Measured p50: about **5.7 ms**.

Block B is **1.9× slower at identical FLOPs.** Attribute the extra 2.7 ms: roughly 0.6 ms to the shuffle (a full-tensor permute, G4), about 0.5 ms to the standalone ReLU and the additive shortcut (two more full-tensor passes, G4), and the remainder to the two-branch fragmentation paying double kernel-launch overhead and running each branch at half-width below the chip's efficient occupancy (G3). The FLOP counter rates A and B as a dead tie. The chip rates B as nearly twice the budget. If you had picked B off a FLOP-Pareto plot — and the decomposed block is exactly the kind of thing that *looks* sophisticated in a paper figure — you would have shipped a model that misses your frame budget by 90%.

![A before and after figure showing that a depthwise convolution has tiny FLOPs that predict it is fast, while its actual cost is reading and writing the full tensor at low arithmetic intensity which makes it bandwidth bound and slow](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-6.png)

### 4.6 Putting the four together: the ShuffleNetV2 block

ShuffleNetV2's block is the four guidelines made flesh. It (G1) keeps the two `1×1` convs at equal channel width; (G2) drops V1's grouped `1×1` convs back to ordinary `1×1`s, accepting slightly more FLOPs for far less memory traffic; (G3) avoids fragmentation — it has a single main path with one cheap split rather than many branches; and (G4) replaces the additive shortcut with a channel split at the start and a concat at the end, eliminating element-wise adds, and keeps a single depthwise in the path. The result outperformed ShuffleNetV1, MobileNetV2, and the FLOP-matched NASNet-A on *measured latency* on both ARM and GPU — not on FLOPs, where some of those rivals looked comparable, but on the wall clock. That is the entire thesis of the paper and of this post: design and *compare* on the direct metric, on the target.

## 5. Measuring the gap yourself: code

Enough theory. Let me show you how to reproduce the gap on your own machine, because seeing it once cures you of FLOP-worship permanently. We will (a) load EfficientNet from `timm`, (b) count its FLOPs, (c) measure its real latency, and (d) build two equal-FLOP blocks — one fragmented, one not — and watch them run far apart.

### 5.1 Load EfficientNet and count FLOPs

`timm` (PyTorch Image Models) is the easiest way to get a correct EfficientNet, and `fvcore` or `thop` will count FLOPs. Here is the FLOP count for B0:

```python
import torch
import timm
from fvcore.nn import FlopCountAnalysis, parameter_count

# EfficientNet-B0 at its native 224x224 input resolution
model = timm.create_model("efficientnet_b0", pretrained=True).eval()
dummy = torch.randn(1, 3, 224, 224)

flops = FlopCountAnalysis(model, dummy)
# fvcore counts MACs (multiply-accumulates); x2 for FLOPs by the common convention
macs = flops.total()
params = parameter_count(model)[""]
print(f"EfficientNet-B0: {macs / 1e9:.2f} GMACs  ({2 * macs / 1e9:.2f} GFLOPs), "
      f"{params / 1e6:.2f}M params")
# -> roughly 0.40 GMACs / 0.80 GFLOPs, 5.29M params
```

A note that trips everyone up: tools disagree about MACs versus FLOPs by a factor of two. `fvcore` reports MACs (it calls them FLOPs, confusingly); the EfficientNet paper's "0.39B FLOPs" for B0 is actually 0.39B MACs. Always check which convention a tool uses before you compare two numbers, or you will "discover" a phantom 2× difference that is pure bookkeeping. This convention confusion is itself a small instance of the post's theme: a number that looks like a measurement is actually a definitional artifact.

### 5.2 Measure real latency the honest way

FLOPs counted, now measure. Latency measurement is full of traps — cold caches, kernel autotuning on the first call, thermal throttling, and CPU frequency scaling all corrupt naive timing. The honest recipe is: warm up, synchronize, time many iterations, report a percentile not a mean. Means hide tail latency; on device you care about p50 and p99.

```python
import time
import numpy as np
import torch

@torch.inference_mode()
def measure_latency(model, dummy, warmup=20, iters=200, device="cpu"):
    model = model.to(device).eval()
    dummy = dummy.to(device)
    # Warm up: triggers lazy init, kernel autotuning, fills caches
    for _ in range(warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()   # GPU work is async; sync before stopping the clock
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    samples = np.array(samples)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p99_ms": float(np.percentile(samples, 99)),
        "mean_ms": float(samples.mean()),
    }

model = timm.create_model("efficientnet_b0", pretrained=True)
dummy = torch.randn(1, 3, 224, 224)
print(measure_latency(model, dummy, device="cpu"))
# On an M2 MacBook CPU, batch=1, expect roughly p50 ~ 18-25 ms, p99 a bit higher.
```

Two details that separate a real measurement from a fake one. First, `torch.cuda.synchronize()` is mandatory on GPU because kernel launches are asynchronous — without it you are timing how fast Python can *queue* work, not how fast the GPU *finishes* it, which is a classic way to "prove" a model is microseconds fast when it is milliseconds slow. Second, **batch size 1.** On-device inference is almost always batch=1 (one camera frame, one user request), and batch=1 is exactly where memory-bound behavior dominates, because you cannot amortize weight loads across a batch. A model that looks efficient at batch=64 on a server can be a disaster at batch=1 on a phone. Always measure at the batch size you will actually ship.

### 5.3 Fragmented vs unfragmented: building the gap

Now the demonstration that matters — two blocks with (approximately) matched FLOPs, one a single fat conv, one fragmented into parallel branches, timed head to head. This is G3 in code.

```python
import torch
import torch.nn as nn

C = 256          # channels in and out
HW = 56          # feature map side
x = torch.randn(1, C, HW, HW)

# Unfragmented: one 3x3 group conv, 8 groups. Few, large ops.
class Unfragmented(nn.Module):
    def __init__(self, c=C, g=8):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Fragmented: split into 8 separate branches, each handling c/8 channels,
# then concat. Same total grouped-conv FLOPs, but 8 small kernels + a concat.
class Fragmented(nn.Module):
    def __init__(self, c=C, branches=8):
        super().__init__()
        cb = c // branches
        self.branches = nn.ModuleList(
            nn.Conv2d(cb, cb, 3, padding=1, bias=False) for _ in range(branches)
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
        self.cb = cb
    def forward(self, x):
        chunks = torch.split(x, self.cb, dim=1)
        outs = [b(c) for b, c in zip(self.branches, chunks)]   # 8 tiny convs
        return self.act(self.bn(torch.cat(outs, dim=1)))       # element-wise concat
```

Both modules compute the same arithmetic: a grouped `3×3` with 8 groups *is* 8 independent `3×3` convs on `c/8` channels each — the FLOP counter reports them as identical. But `Unfragmented` issues one fused grouped-conv kernel; `Fragmented` issues eight tiny kernels plus a split and a concat. Time them with the harness above:

```python
print("unfragmented:", measure_latency(Unfragmented(), x, device="cpu"))
print("fragmented:  ", measure_latency(Fragmented(),   x, device="cpu"))
# Representative result, M2 CPU batch=1:
#   unfragmented: p50 ~ 2.0 ms
#   fragmented:   p50 ~ 3.5-5 ms   (1.7-2.5x slower at identical FLOPs)
# On a GPU the gap is even larger: tiny kernels can't saturate the cores,
# and per-kernel launch overhead dominates.
```

There is your 2×, on your own machine, at matched FLOPs. The fragmented block's extra time is pure $T_{\text{overhead}}$ (eight kernel launches instead of one) plus poor parallel utilization (each tiny conv leaves most of the chip idle) plus the concat's element-wise traffic (G4). None of it shows up in the FLOP count. Run it on a GPU and the gap typically widens, because tiny kernels are even worse at filling thousands of cores than they are at filling a handful of CPU threads.

### 5.4 Watching the correlation break down

One demonstration is an anecdote; a sweep is evidence. To really see FLOPs lie, sweep a family of blocks across a FLOP range, plot measured latency against FLOPs, and look at the scatter. If FLOPs predicted latency, the points would lie on a line. They do not — they smear, and the smear is the gap quantified. Here is a compact sweep that builds blocks at a grid of FLOP targets with varying fragmentation and group counts, measures each, and reports the rank correlation between FLOPs and latency.

```python
import itertools
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

def build_block(channels, groups, branches):
    cb = channels // branches
    convs = nn.ModuleList(
        nn.Conv2d(cb, cb, 3, padding=1, groups=min(groups, cb), bias=False)
        for _ in range(branches)
    )
    bn = nn.BatchNorm2d(channels)
    act = nn.ReLU(inplace=True)

    class Block(nn.Module):
        def forward(self, t):
            chunks = torch.split(t, cb, dim=1)
            outs = [c(p) for c, p in zip(convs, chunks)]
            return act(bn(torch.cat(outs, dim=1)))
    return Block()

x = torch.randn(1, 256, 56, 56)
rows = []
for groups, branches in itertools.product([1, 2, 4, 8], [1, 2, 4, 8]):
    blk = build_block(256, groups, branches).eval()
    macs = FlopCountAnalysis(blk, x).total()
    lat = measure_latency(blk, x, device="cpu")["p50_ms"]
    rows.append((groups, branches, macs / 1e6, lat))

# Rank-correlate FLOPs vs latency across the sweep
macs = np.array([r[2] for r in rows])
lats = np.array([r[3] for r in rows])
order_macs = np.argsort(macs)
order_lats = np.argsort(lats)
# Spearman-style: how often does lower FLOPs mean lower latency?
print("FLOPs (M) and p50 (ms) per (groups, branches):")
for g, b, m, l in sorted(rows, key=lambda r: r[2]):
    print(f"  g={g} br={b}: {m:6.1f} MFLOP -> {l:5.2f} ms")
# Typical finding: the LOWEST-FLOP block (g=8, br=8) is among the SLOWEST,
# while a higher-FLOP unfragmented block (g=1, br=1) is the fastest.
# Spearman rho between FLOPs and latency often lands near 0 or negative.
```

When I run this kind of sweep on a CPU, the punchline is reliable: the *lowest-FLOP* configuration — heavy grouping, heavy fragmentation, `g=8, branches=8` — is consistently among the *slowest*, and the chunky `g=1, branches=1` block, despite the *highest* FLOPs in the grid, is the fastest. The Spearman rank correlation between FLOPs and latency across the grid is frequently near zero, and within a FLOP band it can go negative. That is not noise; it is the four guidelines acting in concert. If you trusted FLOPs to rank these blocks you would pick almost exactly backwards. Save this sweep and run it on every target you deploy to — the *shape* of the smear is a fingerprint of that chip's roofline, and it tells you which design moves will pay off there before you commit to any of them.

## 6. Results: the tables you should actually compare on

Let me assemble the numbers into the comparisons worth keeping. First the EfficientNet family, the canonical demonstration that compound scaling produces a clean accuracy-FLOPs frontier. Figure 7 shows the family at a glance; the full table follows with latency added.

![A matrix figure showing the EfficientNet B0 B3 B5 and B7 models with their FLOP counts parameter counts and ImageNet top-1 accuracy illustrating the climb up the accuracy curve at the cost of an order of magnitude more compute](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-7.png)

### 6.1 EfficientNet B0-B7: FLOPs, params, accuracy, latency

These are the published EfficientNet figures (Tan & Le, 2019) for FLOPs (reported as MACs in the paper's "B" units), parameters, and ImageNet top-1. The input resolution rises with the model. The latency column is *representative* batch=1 inference on a mid-range mobile-class CPU and should be read as order-of-magnitude, not a benchmark — actual numbers depend heavily on the runtime, the chip, and whether the depthwise kernels are well-optimized for the target.

| Model | Resolution | FLOPs (B MACs) | Params (M) | Top-1 (%) | Approx. CPU latency (ms, batch=1) |
| --- | --- | --- | --- | --- | --- |
| B0 | 224 | 0.39 | 5.3 | 77.1 | ~20 |
| B1 | 240 | 0.70 | 7.8 | 79.1 | ~32 |
| B2 | 260 | 1.0 | 9.2 | 80.1 | ~42 |
| B3 | 300 | 1.8 | 12 | 81.6 | ~70 |
| B4 | 380 | 4.2 | 19 | 82.9 | ~150 |
| B5 | 456 | 9.9 | 30 | 83.6 | ~310 |
| B6 | 528 | 19 | 43 | 84.0 | ~560 |
| B7 | 600 | 37 | 66 | 84.3 | ~1000 |

Read the curve, not the rows. Going B0 to B7 buys you about **7 points of top-1** (77.1 to 84.3) for roughly **95× more FLOPs** (0.39 to 37) and **50× the latency**. The marginal point of accuracy gets brutally expensive at the top. On the edge you almost never want B5 and up — the accuracy is lovely and the latency is a deal-breaker. B0 to B2 is the sweet spot for phones; even B0 is often too slow for real-time video and you reach for a latency-searched MobileNetV3 or a quantized B0 instead. Notice too that the *latency* climbs faster than FLOPs at the high end — the big models run at higher resolution, and the depthwise convs at high resolution are increasingly memory-bound, so each FLOP is getting slower. The FLOP-to-latency ratio is not constant even *within* one well-designed family.

### 6.2 The headline: equal FLOPs, unequal latency

This is the table that should be tattooed on every architecture reviewer's forearm. It collects efficient blocks at *roughly matched FLOPs* and shows how far their measured latencies spread. Numbers are representative of the patterns the ShuffleNetV2 paper and subsequent mobile-CV benchmarks report; treat them as directional, with the *relative ordering* being the robust, reproducible part.

| Block / model (matched ~ FLOP class) | FLOPs | Why FLOPs mislead | Relative latency (batch=1) |
| --- | --- | --- | --- |
| Single fat conv (compute-bound) | 1.0× | high arithmetic intensity, fills the cores | 1.0× (fastest) |
| ShuffleNetV2-style block | 1.0× | follows G1-G4, low traffic, unfragmented | ~1.0-1.2× |
| MobileNetV2 inverted bottleneck | 1.0× | depthwise is memory-bound; add is element-wise | ~1.3-1.5× |
| Grouped conv, g=8 (V1-style) | 1.0× | G2: extra channels inflate memory traffic | ~1.5-1.8× |
| Fragmented multi-branch (NAS-found) | 1.0× | G3: kernel overhead + poor parallelism | ~1.8-3.0× (slowest) |

Same FLOP column, a 2-3× spread in the latency column. That spread is the gap. Every entry is "efficient" by the FLOP count; only the top two are efficient on the chip. If you select architectures by the first column you will systematically pick from the bottom of the second.

### 6.3 The four guidelines, condensed

For quick reference, the four ShuffleNetV2 guidelines as a decision aid — the same content as figure 5 in prose-table form, because you will want to grep it later:

| Guideline | The rule | The mechanism | Designed-in fix |
| --- | --- | --- | --- |
| G1 | Keep input channels = output channels | MAC is minimized at 1:1 for fixed FLOPs (AM-GM) | Equal-width `1×1` convs |
| G2 | Do not over-use grouped convolution | More groups inflate activation memory traffic per FLOP | Moderate or no grouping |
| G3 | Avoid network fragmentation | Many small ops kill parallelism, add launch overhead | Few, large branches |
| G4 | Reduce element-wise operations | ReLU, add, copy cost real bytes, ~zero FLOPs | Fuse ops; concat over add |

## 7. The deeper truth: this is a roofline argument

Everything above is one idea seen from four angles, and the idea is the roofline. Plot achievable throughput (FLOP/s) on the y-axis against arithmetic intensity (FLOPs per byte) on the x-axis and you get a line that rises with intensity until it hits a ceiling — the chip's peak compute — and then flattens. The rising part is the **bandwidth roof**: at low intensity you are limited by how fast memory feeds the compute units, and throughput grows linearly with intensity. The flat part is the **compute roof**: at high intensity the units are saturated and more intensity buys nothing. Where an operation sits on this plot, set by its arithmetic intensity, decides whether reducing its FLOPs helps at all. The roofline post in this series, [the roofline model where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), derives all of this carefully; here I want to connect it to the architecture choices.

A dense `3×3` conv with many channels has high arithmetic intensity — it reuses each loaded input value across many output channels and each weight across many spatial positions. It lives on the flat compute roof. For an op on the compute roof, latency is genuinely $\text{FLOPs}/\text{peak}$, so reducing FLOPs *does* reduce latency. FLOPs do not lie for compute-bound ops.

A depthwise conv has terrible arithmetic intensity — 9 MACs per output element, no channel reuse, so it streams the whole tensor through for a trickle of math. It lives way down on the bandwidth roof. For an op on the bandwidth roof, latency is $\text{bytes}/\text{bandwidth}$, *independent of FLOPs*. Halving the FLOPs of a depthwise conv (say by halving the kernel area) barely moves its latency, because it still has to read and write the same activation tensor. This is the formal statement of G4 and of section 4.1's inequality. The "efficient" blocks of section 3 are efficient precisely by trading dense, compute-bound convolutions for cheap, *memory-bound* ones — which is a great trade when you are FLOP-limited and a *bad* trade when you are bandwidth-limited, and most edge chips are bandwidth-limited at batch=1.

#### Worked example: a roofline verdict on two blocks

Take a chip with peak 100 GFLOP/s of compute and 10 GB/s of memory bandwidth — a plausible small NPU. Its roofline "ridge point" (where the bandwidth roof meets the compute roof) is at arithmetic intensity $100 \text{ GFLOP/s} \div 10 \text{ GB/s} = 10$ FLOPs per byte. Any op below 10 FLOPs/byte is memory-bound; above it, compute-bound.

- A `1×1` conv with 256 in and 256 out channels on a `14×14` map: FLOPs $\approx 2\cdot 196\cdot 256\cdot 256 \approx 25.7$ MFLOP; bytes (fp16, 2 B each) $\approx 2\cdot(196\cdot256 + 196\cdot256 + 256\cdot256) \approx 2\cdot 165{,}888 \approx 332$ KB. Intensity $\approx 25.7\text{M} / 332\text{K} \approx 77$ FLOPs/byte. That is well above the ridge point of 10 — **compute-bound**. Reducing its FLOPs helps.
- A depthwise `3×3` conv on the same `14×14×256` map: FLOPs $\approx 2\cdot 196\cdot 256\cdot 9 \approx 0.9$ MFLOP; bytes $\approx 2\cdot(196\cdot256 + 196\cdot256) \approx 2\cdot 100{,}352 \approx 201$ KB. Intensity $\approx 0.9\text{M} / 201\text{K} \approx 4.5$ FLOPs/byte. Below the ridge point — **memory-bound**. Its latency is set by 201 KB / 10 GB/s $\approx 20$ µs, and *cutting its FLOPs does nothing*, because the 20 µs is all memory.

So on this chip, the `1×1` conv (25× more FLOPs than the depthwise) might actually be *faster per FLOP delivered*, and the depthwise conv — the "efficient" op — is the one stuck on the bandwidth roof. A FLOP-minimizing design would lean hard on depthwise convs and land in exactly the slow region. This is not a quirk of one chip; it is the structural reason FLOP-optimal mobile nets underperform on real silicon.

### 7.1 Why batch=1 makes everything worse

There is a structural reason the edge suffers this more than the cloud, and it is worth deriving because it explains why the *same architecture* can be compute-bound on a server and memory-bound on a phone. The lever is **batching**, and it works through arithmetic intensity.

Consider a `1×1` conv as a matrix multiply: weights $W$ of shape $C_{out}\times C_{in}$ times an activation matrix $X$ of shape $C_{in}\times N$, where $N = H\cdot W\cdot \text{batch}$ is the number of spatial-times-batch positions. The FLOPs are $\propto C_{out}\cdot C_{in}\cdot N$. The bytes you must move are: the weights $C_{out}\cdot C_{in}$ (loaded once), the input $C_{in}\cdot N$, and the output $C_{out}\cdot N$. The arithmetic intensity is therefore

$$
\text{intensity} = \frac{C_{out} C_{in} N}{C_{out} C_{in} + (C_{in}+C_{out})N}.
$$

Look at the two regimes. When $N$ is small (small feature map, batch=1) the weight term $C_{out}C_{in}$ in the denominator dominates and the intensity is *low* — you pay full price to load the whole weight matrix and use it for only a few positions, so you are bandwidth-bound. When $N$ is large (big feature map, or a fat batch on a server) the $N$ terms dominate top and bottom, the weight-load cost amortizes across many positions, and the intensity rises toward its ceiling $\frac{C_{out}C_{in}}{C_{in}+C_{out}}$ — you become compute-bound. Batching is, mathematically, a *weight-reuse multiplier*: it raises arithmetic intensity by spreading each loaded weight across more positions, sliding the operation rightward across the roofline from the bandwidth roof toward the compute roof.

This is why FLOPs are a *fair* proxy on a cloud GPU serving batch=64 and a *lying* proxy on a phone serving batch=1. The exact same MBConv block, with the exact same FLOPs, sits on the compute roof in the datacenter and on the bandwidth roof in your pocket. An architecture selected by its server-batch FLOP efficiency is being selected on a roofline regime it will never see on device. It also explains why depthwise convolutions are *especially* edge-hostile: a depthwise conv has no channel-mixing inner product to amortize weights against in the first place, so batching helps it far less than it helps a dense `1×1`. The depthwise op stays memory-bound across almost the entire useful batch range. The "efficient" op is the one that benefits least from the cloud's favorite trick.

#### Worked example: the same block, two batch sizes

Take a `1×1` conv with $C_{in}=C_{out}=256$ on a `14×14` map ($H\cdot W = 196$), fp16. At **batch=1**, $N=196$: intensity $= \frac{256\cdot256\cdot196}{256\cdot256 + 512\cdot196} = \frac{12.85\text{M}}{65{,}536 + 100{,}352} = \frac{12.85\text{M}}{165{,}888} \approx 77$ FLOPs/byte. At **batch=32**, $N=6272$: intensity $= \frac{256\cdot256\cdot6272}{65{,}536 + 512\cdot6272} = \frac{411\text{M}}{65{,}536 + 3.21\text{M}} \approx 126$ FLOPs/byte. Batching pushed the intensity up about 1.6× here, and for blocks where the weight term dominates more strongly the swing is larger. On our example NPU (ridge point 10 FLOPs/byte) this particular op clears the ridge at both batch sizes, but a *grouped* or *depthwise*-heavy block with intrinsically low intensity is exactly the one that crosses from memory-bound at batch=1 to compute-bound only at large batch — which is to say, never, on the edge.

## 8. Case studies: where this has actually bitten teams

Three real instances from the literature and from shipped systems, because the abstract argument is most convincing when you see it cash out.

**ShuffleNetV2 vs NASNet-A on a phone.** NASNet-A was found by a FLOP-and-accuracy NAS and looks excellent on the accuracy-FLOPs Pareto plot. On an ARM CPU at matched FLOPs, ShuffleNetV2 ran considerably faster — NASNet-A's many-branch cells are exactly the G3 fragmentation that wrecks mobile latency. The FLOP-optimal architecture lost to the latency-aware one on the metric that ships. This is the original paper's own headline comparison and it has held up across follow-on benchmarks.

**MobileNetV3 vs MobileNetV2 on a Pixel.** MobileNetV3 (Howard et al., 2019) was tuned with *hardware-aware* NAS (platform-aware NetAdapt) that searched on **measured Pixel latency**, not FLOPs. The result: MobileNetV3-Large hit higher ImageNet accuracy than V2 at *lower measured latency on the Pixel*, even though a pure FLOP comparison would have rated the gain as modest. Among the changes the latency search discovered were redesigning the expensive early and final layers — places where FLOPs were low but latency was high because of resolution and memory effects. The lesson, again: search on the direct metric and the architecture comes out different and faster.

**EfficientNet on a GPU vs the edge.** EfficientNet dominates the accuracy-FLOPs frontier and trains and serves beautifully on cloud GPUs where compute is the bottleneck and FLOPs are a fair proxy. But teams who took B0 to phones often found its dense use of depthwise-separable convs at multiple resolutions made it *slower than its FLOP count promised*, sometimes slower than a hand-tuned MobileNetV3 with more FLOPs. The follow-up, EfficientNet-Lite, explicitly removed the squeeze-and-excite blocks and swapped the swish activation for ReLU6 — both are element-wise-heavy, G4-violating ops that are cheap in FLOPs and expensive in mobile latency and quantize poorly. The "Lite" variants gave up a little accuracy for a real on-device speedup. That redesign *is* the four guidelines applied as a patch.

**MnasNet and FBNet: latency in the loss, by lookup table.** MnasNet (Tan et al., 2019) was the first to put *measured device latency* directly into the multi-objective NAS reward, running candidate models on real Pixel phones during the search. FBNet (Wu et al., 2019) made this practical at scale by replacing on-device measurement with a *latency lookup table* — precompute the measured latency of each candidate operator at each shape on the target, then sum the table entries during a differentiable search, so the search is latency-aware without timing every candidate live. Both produced models that beat FLOP-matched, hand-designed nets on measured mobile latency, and both rediscovered the four guidelines without being told them — their searched blocks are unfragmented, channel-balanced, and light on element-wise ops, because the latency signal punished anything else. That is the strongest possible evidence that the guidelines are real and not folklore: an optimizer that only ever saw latency converged to them on its own.

The common thread: every time a model was selected or scaled by FLOPs and then deployed, the on-device latency surprised someone. Every time the design loop measured latency on the target, the surprise went away — and the winning architecture looked different, usually chunkier and less elegant, than the FLOP-optimal one.

## 9. The takeaway for edge design: search on latency, not FLOPs

If FLOPs mislead, the obvious fix is to stop using them as the search objective and use measured latency on the target device instead. That is precisely what **hardware-aware neural architecture search** does, and it is why the field moved there. MnasNet, MobileNetV3, FBNet, and ProxylessNAS all put a latency term — often a latency *lookup table* indexed by the op and its shape, calibrated by measuring on the real device — directly into the search objective. The search then learns, on its own, to obey the four guidelines: it discovers that fragmented blocks are slow, that lopsided channel ratios cost bandwidth, that element-wise ops add up, because the latency signal punishes them. The result is architectures that are Pareto-optimal in *latency* versus accuracy, which is the frontier you actually deploy on. This is its own deep topic, covered in [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas); the connection to make here is that hardware-aware NAS exists *because FLOPs lie* — it is the field's institutional response to this exact gap.

So when do you reach for which tool? Figure 8 is the decision, and it forks on one question: what kind of silicon ships the model?

![A decision tree figure forking on whether the deployment target is a compute-bound cloud GPU where FLOPs are an acceptable proxy or a memory-bound edge chip where you must measure latency on device and use ShuffleNetV2 rules or hardware-aware NAS](/imgs/blogs/efficientnet-shufflenet-and-the-flops-latency-gap-8.png)

### 9.1 When EfficientNet/ShuffleNet fit, and when a latency-searched net wins

**Reach for EfficientNet (compound scaling)** when your target is a fat GPU or TPU — cloud serving, a workstation, an autonomous-vehicle compute box with a discrete GPU — where compute is plentiful, you batch requests, and FLOPs are a fair proxy for cost. Compound scaling gives you a principled dial to trade accuracy for compute, and the accuracy-per-FLOP is excellent. It is also a fine *starting point* for the edge if you take B0 (or EfficientNet-Lite), quantize it, and *then measure*. Just do not assume the FLOP-optimal point is the latency-optimal point.

**Reach for ShuffleNet-style hand design (the four guidelines)** when you are designing a block by hand for a mobile CPU or a known NPU and you want a fast, robust default without running a full NAS. The guidelines are cheap to apply, they generalize across chips reasonably well, and they will keep you off the worst latency cliffs. ShuffleNetV2 and MobileNetV2/V3 blocks are battle-tested mobile primitives.

**Reach for a latency-searched net (MobileNetV3, FBNet, or your own hardware-aware NAS)** when latency is the hard constraint, you have the target device in hand to measure on, and the last 10-20% of latency matters enough to justify the search cost. This is the right move for real-time on-device vision (camera pipelines, AR), where you must hit a frame budget on a specific phone. A latency-searched net tuned to *your* chip will beat a generic efficient net on *your* chip, because it has internalized that chip's roofline.

**Do not** select or compare architectures on FLOPs alone for any edge target. Do not trust a FLOP-Pareto plot to predict a latency-Pareto plot. Do not assume an op with fewer FLOPs is faster — check whether it is compute-bound or memory-bound first. And do not optimize a memory-bound op's FLOPs at all; optimize its memory traffic (fuse it, keep tensors on-chip, raise its arithmetic intensity by giving it more channel reuse).

The choice condenses into a small table you can keep on a sticky note:

| Situation | Target | Best tool | Why |
| --- | --- | --- | --- |
| Cloud serving, batched | GPU / TPU | EfficientNet, compound scaling | compute-bound; FLOPs are a fair proxy; clean accuracy dial |
| Mobile, hand-designed, no NAS budget | mobile CPU / NPU | ShuffleNetV2 / MobileNetV2 blocks | four guidelines keep you off latency cliffs cheaply |
| Mobile, hard frame budget, device in hand | a specific phone | MobileNetV3 / FBNet / hardware-aware NAS | searches on measured latency; beats generic nets on that chip |
| Edge, already have a net, over budget | the deployed device | quantize + fuse + profile, re-measure | int8 raises intensity; fusion cuts G4 traffic; cheapest wins first |
| Microcontroller, KB-scale | Cortex-M class | latency/memory-searched tiny net (MCUNet-style) | SRAM and Flash budget dominate; FLOPs are nearly irrelevant |

### 9.2 Stress-testing the recommendation

What breaks this advice? A few honest edge cases. If your edge target has *unusually high* memory bandwidth relative to compute (some NPUs with large on-chip SRAM keep activations resident, raising effective intensity), the depthwise/memory-bound penalty shrinks and FLOP-efficient designs do better than the roofline-pessimist would predict — measure, do not assume the penalty. If you quantize to int8, arithmetic intensity *rises* (you move half the bytes per FLOP versus fp16), which shifts memory-bound ops rightward toward the compute roof and can rescue an otherwise bandwidth-bound block — quantization and architecture choice interact, which is why this series keeps insisting you re-profile after every lever. And if the op you care about falls back to CPU because the NPU does not support it (depthwise convs and channel shuffles are common fallback victims on immature delegates), all bets are off — the fallback transfer alone can dominate, and a "more efficient" block that triggers a fallback is far slower than a chunky one that stays on the accelerator. The meta-lesson is unchanged: the only verdict that counts is the one the target device gives you.

## 10. How I would actually do this on a deadline

Tying it to a workflow, because the abstract advice needs a procedure. Suppose you are handed a classification model to ship on a specific Android phone with a frame budget of 30 ms.

1. **Start from a sane baseline**, not a NAS run. EfficientNet-Lite0 or MobileNetV3-Large, pretrained. These already obey most of the four guidelines and quantize cleanly.
2. **Measure first, on the actual phone**, batch=1, with warm-up and percentiles, using the runtime you will ship (LiteRT/TFLite with the NNAPI or GPU delegate, or whatever your target supports). Get a p50 and a p99. This is your ground truth; ignore the FLOP count.
3. **Profile the per-op latency** on device. Most runtimes expose an op-level profiler. Find the ops eating the budget. They will usually be the memory-bound ones — depthwise convs at high resolution, the shuffles, the squeeze-and-excite element-wise ops — exactly the G4/G1 offenders, *not* the FLOP-heavy ones.
4. **Fix the actual bottleneck.** Fuse what you can (let the converter fold activations and bias). Drop or simplify the element-wise-heavy modules (squeeze-and-excite, exotic activations) if they cost more latency than they buy accuracy. Re-balance channel widths toward 1:1 where you have a lopsided `1×1`. Reduce input resolution if a high-res early stage is memory-bound — often a bigger latency win than any block swap.
5. **Quantize and re-measure**, because int8 changes the roofline and may turn a memory-bound block compute-bound (see [a-taxonomy-of-model-compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for how the levers compose).
6. **Only then, if still over budget, reach for hardware-aware NAS** tuned to this phone. It is the most expensive option; spend it last, when the cheap structural fixes are exhausted.

Notice that not one step in that loop optimizes FLOPs. Every step optimizes measured latency on the target. That is the whole discipline, and it is the only thing that reliably closes the gap between the slide and the phone.

## 11. Key takeaways

- **FLOPs measure arithmetic; latency is set by whichever of compute, memory, and overhead is the bottleneck.** On a memory-bound op, FLOPs do not appear in the latency at all. Two equal-FLOP blocks can differ 2-3× on one chip.
- **EfficientNet's compound scaling grows depth, width, and resolution together** by a shared coefficient $\phi$, under $\alpha\cdot\beta^2\cdot\gamma^2\approx 2$. The squares and the 2 fall directly out of FLOPs scaling as $d\cdot w^2\cdot r^2$, so each $\phi$ step doubles FLOPs. Balanced scaling beats single-dimension scaling by 2-3 points at matched FLOPs because each dimension's ceiling is set by the other two.
- **ShuffleNet's grouped conv cuts channel-mixing FLOPs; the channel shuffle restores cross-group information at zero FLOPs** — but zero FLOPs is not zero cost, because the shuffle still moves the whole tensor.
- **The four ShuffleNetV2 guidelines name where FLOPs and latency diverge**: G1 equal channel width minimizes memory access cost (AM-GM, minimized at 1:1); G2 excessive grouping inflates traffic; G3 fragmentation kills parallelism (measured ~3× on GPU); G4 element-wise ops cost real bytes (measured ~20% from ReLU + add alone).
- **Depthwise convolutions are bandwidth-bound** — low arithmetic intensity, so they sit on the bandwidth roof where cutting FLOPs does nothing. "Efficient" blocks trade compute-bound convs for memory-bound ones, which helps only when you were compute-limited.
- **Compound scaling and FLOP comparison are valid on compute-bound cloud GPUs;** on memory-bound edge chips at batch=1, FLOPs systematically mislead.
- **Hardware-aware NAS searches on measured latency precisely because FLOPs lie** — and it rediscovers the four guidelines on its own. MobileNetV3 beat V2 by searching on Pixel latency, not FLOPs.
- **The only verdict that counts is the target device's.** Measure batch=1, warm up, report percentiles, profile per-op, fix the memory-bound bottleneck, and re-profile after every change.

## 12. Further reading

- **Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML 2019** — the compound-scaling paper; read section 3 for the constraint derivation and figure 3 for the single-vs-compound comparison.
- **Ma, Zhang, Zheng & Sun, "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," ECCV 2018** — the four guidelines, with the measured experiments behind each one. The most important paper in this post.
- **Zhang, Zhou, Lin & Sun, "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices," CVPR 2018** — grouped convolution plus the original channel-shuffle trick.
- **Han et al., "GhostNet: More Features from Cheap Operations," CVPR 2020** — ghost feature maps from cheap linear ops.
- **Howard et al., "Searching for MobileNetV3," ICCV 2019** — hardware-aware NAS on measured Pixel latency; the case study in section 8.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame; [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) and [the roofline model where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for the measurement discipline this post rests on; [building blocks for efficient models](/blog/machine-learning/edge-ai/building-blocks-for-efficient-models) and [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) for the block primitives; [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas) for searching on latency; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that stitches every lever together.
