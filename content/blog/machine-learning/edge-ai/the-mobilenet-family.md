---
title: "The MobileNet family: V1 to V3, multipliers, and latency-driven design"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Trace MobileNet from V1's depthwise-separable backbone through V2's inverted residuals to V3's NAS-tuned, h-swish design — and walk away able to pick, scale, and quantize the right variant for your device."
tags:
  [
    "edge-ai",
    "model-optimization",
    "mobilenet",
    "efficient-architecture",
    "depthwise-separable",
    "neural-architecture-search",
    "quantization",
    "inference",
    "efficient-ml",
    "computer-vision",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-mobilenet-family-1.png"
---

There is a moment every engineer who ships vision to phones eventually lives through. You trained a clean ResNet-50, it hits 76% top-1 on ImageNet, everyone is happy, and then you put it on a mid-range Android phone and the camera preview turns into a slideshow. Each frame takes 180 ms on the CPU, the phone gets hot, the battery graph nosedives, and the product manager asks why the "AI" feature feels broken. You did nothing wrong in the cloud. You just used a model designed for a world with infinite FLOPs and a fat memory bus, and the phone has neither.

MobileNet is the family of networks built specifically for that world. It is the reference design for on-device convolutional networks — the thing you reach for first, the baseline everyone else is measured against, and the model that taught the field a sequence of lessons about getting more accuracy per millisecond. There are three generations, and each one solved a different piece of the puzzle. V1 (2017) replaced the expensive standard convolution with a depthwise-separable factorization and gave you two dials — a width multiplier and a resolution multiplier — to slide along an accuracy-versus-compute curve. V2 (2018) reshaped the building block into an inverted residual with a linear bottleneck, which both raised accuracy and made inference far gentler on memory. V3 (2019) stopped hand-designing the network at all: it let a platform-aware neural architecture search optimize directly for measured latency on a real phone, then bolted on a cheap activation (h-swish) and squeeze-and-excite attention, and hand-fixed the parts the search left expensive.

This post traces that evolution and, more importantly, the *reasoning* behind it. We will derive how the width multiplier $\alpha$ and resolution multiplier $\rho$ each scale compute roughly quadratically, so a single base model becomes a whole family on one curve. We will see why the expand-then-project shape of V2's inverted residual is exactly what a depthwise convolution wants. We will derive h-swish from swish and show precisely why dropping the exponential matters on integer hardware. And we will close the loop with runnable code: loading V2 and V3 from `torchvision`, applying a width multiplier, quantizing to int8 (MobileNet is the canonical quantization-friendly network), and benchmarking honestly on CPU. The whole arc sits on the series' four-lever frame — quantization, pruning, distillation, and efficient architecture — and MobileNet is the purest example of that fourth lever: when you redesign the architecture itself, you move the entire [accuracy-efficiency Pareto frontier](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), not just slide along it.

![Timeline of the three MobileNet generations from V1 depthwise-separable through V2 inverted residuals to V3 NAS-tuned design](/imgs/blogs/the-mobilenet-family-1.png)

There is a reason this one family deserves a whole post rather than a paragraph. MobileNet is where the field learned, in public and with measured numbers, that *architecture* is a first-class optimization lever — that you do not have to take a model as given and only compress it afterward, you can design the model so that it is fast from the start. Every later efficient network, from EfficientNet to the mobile vision transformers, traces a line back to the depthwise-separable block and the inverted residual. And the three generations are a compressed history of how the field's thinking matured: from "factor the expensive operation" (V1), to "reshape the block around the cheap operation" (V2), to "stop guessing and let a search optimize the metric that actually matters on the device" (V3). If you understand why each step followed the last, you understand most of what there is to know about designing for the edge.

By the end you will be able to look at a device latency budget and say, with numbers, which MobileNet variant to start from, how to scale it, and how to squeeze it to int8 without losing accuracy you cannot afford. Let us start where the whole family started: the convolution that costs too much.

## Why a standard convolution is too expensive

A standard convolution is doing two jobs at once, and that is the whole problem. When a $3\times3$ convolution with $C_{in}$ input channels produces $C_{out}$ output channels, every output channel is a weighted sum over a $3\times3$ spatial window across *all* $C_{in}$ input channels. It is simultaneously filtering each location spatially and mixing information across channels. Bundling those two jobs into one operation is convenient mathematically, but it is enormously expensive.

Let us count. For an input feature map of spatial size $H \times W$, a standard convolution with kernel size $K \times K$ costs

$$
\text{FLOPs}_{\text{std}} = H \cdot W \cdot C_{in} \cdot C_{out} \cdot K^2
$$

multiply-accumulates (we will count a multiply-accumulate as one operation throughout; some papers double it). The cost is the product of five things, and the painful ones are $C_{in} \cdot C_{out}$ — quadratic in channel count — multiplied by $K^2$. A layer with 256 input and 256 output channels at $K=3$ on a $14\times14$ map costs $14 \cdot 14 \cdot 256 \cdot 256 \cdot 9 \approx 1.15 \times 10^9$ operations. One layer. A network has dozens.

This is the building block MobileNetV1 replaced, and to understand the replacement you need the [building blocks of efficient models](/blog/machine-learning/edge-ai/building-blocks-for-efficient-models) — depthwise and pointwise convolution — which that companion post derives from scratch. I will summarize just enough here to make the multiplier math land, then build on it.

### The depthwise-separable factorization

The depthwise-separable convolution splits the bundled operation into the two jobs it was secretly doing:

1. A **depthwise convolution** filters each input channel independently. There is one $K \times K$ spatial filter per channel, and it does not mix channels at all. Output has the same number of channels as input.
2. A **pointwise convolution** — a $1\times1$ convolution — then mixes across channels, taking the $C_{in}$ depthwise outputs and producing $C_{out}$ channels, with no spatial extent.

Together they accomplish the same thing as a standard convolution: spatial filtering plus channel mixing. The cost, though, is dramatically lower. The depthwise step costs

$$
\text{FLOPs}_{\text{dw}} = H \cdot W \cdot C_{in} \cdot K^2
$$

(note: no $C_{out}$ — each channel is filtered independently), and the pointwise step costs

$$
\text{FLOPs}_{\text{pw}} = H \cdot W \cdot C_{in} \cdot C_{out}
$$

(note: no $K^2$ — it is a $1\times1$). The ratio of the factorized cost to the standard cost is

$$
\frac{\text{FLOPs}_{\text{dw}} + \text{FLOPs}_{\text{pw}}}{\text{FLOPs}_{\text{std}}}
= \frac{H W C_{in} K^2 + H W C_{in} C_{out}}{H W C_{in} C_{out} K^2}
= \frac{1}{C_{out}} + \frac{1}{K^2}.
$$

For $K=3$ and a typical $C_{out}$ in the hundreds, $1/C_{out}$ is tiny and the ratio is dominated by $1/K^2 = 1/9$. So the depthwise-separable convolution does roughly the same job for about **8 to 9 times fewer operations**. That single factorization is the entire foundation MobileNetV1 stands on. Everything after it is about spending that saving more wisely.

A subtlety worth flagging now, because it haunts the rest of the story: the pointwise $1\times1$ convolution is where most of the compute *and* most of the parameters now live. After factorization, the depthwise filters are cheap and almost free in parameters; the $1\times1$ mixers carry the load. In the V1 paper's own breakdown, the $1\times1$ convolutions account for about 95% of the multiply-adds and about 75% of the parameters. When V2 and V3 optimize the block, they are mostly optimizing the shape and placement of those $1\times1$ convolutions. Keep that in mind.

### The science: why the depthwise step is memory-bound

Here is the part the FLOP count hides, and it is the single most important idea for reasoning about MobileNet's real-world speed. A depthwise convolution does almost no arithmetic per byte of data it moves. The relevant quantity is *arithmetic intensity* — the ratio of compute operations to bytes of memory traffic, in FLOPs per byte. The roofline model says that when arithmetic intensity is low, you are *memory-bound*: the processor sits idle waiting for data, and your latency is set by memory bandwidth, not by how many FLOPs you have.

Compute the arithmetic intensity of a depthwise convolution. For a single channel of an $H \times W$ feature map with a $K \times K$ filter, the work is $H W K^2$ multiply-adds, and the data touched is about $H W$ input elements, $H W$ output elements, and $K^2$ filter weights. In bytes (call element size $s$), intensity is roughly

$$
I_{\text{dw}} = \frac{H W K^2}{s \cdot (2 H W + K^2)} \approx \frac{K^2}{2s}
$$

for $H W \gg K^2$. With $K=3$ and fp32 ($s=4$), that is about $9/8 \approx 1.1$ FLOPs per byte. That is *terrible* — a modern mobile CPU might want tens of FLOPs per byte to stay compute-bound. Now compute it for a $1\times1$ pointwise convolution producing $C_{out}$ channels from $C_{in}$:

$$
I_{\text{pw}} = \frac{H W C_{in} C_{out}}{s \cdot (H W C_{in} + H W C_{out} + C_{in} C_{out})}.
$$

When $C_{in}, C_{out}$ are in the hundreds, the $C_{in} C_{out}$ weight-reuse term dominates the denominator favorably and intensity climbs to tens of FLOPs per byte — the $1\times1$ is essentially a matrix multiply and reuses each weight across all $H W$ spatial positions. So within one MobileNet block, the depthwise step is memory-bound and the pointwise step is compute-bound. They live on opposite sides of the roofline.

This is *why* a MobileNet with one-ninth the FLOPs of a standard-convolution network does not run nine times faster: a large fraction of its remaining work is the memory-bound depthwise step, which the FLOP count makes look free but which the memory system makes expensive. It is also why V3's measured-latency NAS was the right move — only a latency measurement on real silicon captures this. The [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) post derives this framework in full; the takeaway here is that MobileNet is the poster child for FLOPs being a leaky proxy for latency, and every design decision after V1 was, in part, a response to that gap.

## MobileNetV1: the backbone and two dials

MobileNetV1 (Howard et al., 2017) is almost embarrassingly simple in hindsight. Take a straight stack of depthwise-separable blocks — each block is a $3\times3$ depthwise convolution, batch norm, ReLU, then a $1\times1$ pointwise convolution, batch norm, ReLU — with strided depthwise convolutions to downsample, and a global average pool and fully connected classifier at the end. No residual connections, no fancy blocks, no attention. The full network at the standard $224\times224$ input is about 4.2 million parameters and 569 million multiply-adds, and it reaches 70.6% top-1 on ImageNet. For comparison, a VGG-16 of that era used about 138 million parameters and 15 billion multiply-adds to reach 71%. MobileNetV1 got within a fraction of a point at one-thirtieth the parameters and one-twenty-fifth the compute.

The full V1 body is 13 depthwise-separable blocks plus an initial standard $3\times3$ convolution (which is kept standard because at the input there are only 3 channels, so factorization buys little). Channel counts double at each downsampling stage — 32, 64, 128, 256, 512, 1024 — and the spatial resolution halves, the familiar pyramid shape. The standard initial conv is the one place V1 keeps a full convolution; everything after it is depthwise-separable. That single design fact is worth internalizing, because it recurs in V2 and V3: the *first* layer stays cheap-to-factorize-resistant and full, and the factorized blocks do the bulk of the work downstream.

But the lasting contribution of V1 was not the backbone. It was the recognition that a single network is the wrong deliverable. Different devices have different budgets, so V1 shipped *two scalar knobs* that let you generate a whole spectrum of models from one design, and it characterized how each knob trades accuracy for cost. Those knobs are the width multiplier $\alpha$ and the resolution multiplier $\rho$. The genuinely novel move was not the knobs themselves — scaling channels is old — but *publishing the trade-off curve*: the paper measured accuracy at each $(\alpha, \rho)$ so a practitioner could pick a point without retraining from scratch to discover it. That turned model selection from a research project into a lookup.

A limitation that V2 would later fix: V1 has no residual connections. A plain stack of depthwise-separable blocks is a long chain with no skip path, so gradients have a harder time flowing during training and the effective depth is limited — pushing V1 much deeper yields diminishing returns. That structural gap, more than any single number, is what motivated the next generation's redesign of the block.

### The width multiplier alpha

The width multiplier $\alpha$ uniformly thins the network. At a layer that would have had $C_{in}$ input and $C_{out}$ output channels, you instead use $\alpha C_{in}$ and $\alpha C_{out}$. Typical values are $\alpha \in \{1.0, 0.75, 0.5, 0.35, 0.25\}$. The base model is $\alpha = 1.0$; everything smaller is a thinned version.

Now derive the effect on compute. The dominant cost in a V1 block is the pointwise convolution, $\text{FLOPs}_{\text{pw}} = H W C_{in} C_{out}$. Replace $C_{in} \to \alpha C_{in}$ and $C_{out} \to \alpha C_{out}$:

$$
\text{FLOPs}_{\text{pw}}(\alpha) = H W (\alpha C_{in})(\alpha C_{out}) = \alpha^2 \cdot H W C_{in} C_{out}.
$$

The pointwise cost scales as $\alpha^2$ because *both* the input and output channel counts shrink by $\alpha$. The depthwise cost scales only as $\alpha$ (it touches $\alpha C_{in}$ channels but no $C_{out}$), but since pointwise dominates, the network-level compute scales **approximately quadratically in $\alpha$**. Parameters scale the same way, since the $1\times1$ weight tensor is $C_{in} \times C_{out}$. So $\alpha = 0.5$ gives roughly $0.5^2 = 0.25$ of the compute — a 4x reduction — at the cost of some accuracy. The V1 paper reports $\alpha = 1.0$ at 70.6%, $\alpha = 0.75$ at 68.4%, $\alpha = 0.5$ at 63.7%, and $\alpha = 0.25$ at 50.6%. The accuracy falls smoothly and predictably, which is exactly what makes $\alpha$ useful: you can pick a compute budget and read off the expected accuracy.

### The resolution multiplier rho

The resolution multiplier $\rho$ scales the input image. Instead of $224 \times 224$, you feed $\rho \cdot 224 \times \rho \cdot 224$, with $\rho$ implied by common choices of input resolution: 224, 192, 160, 128, corresponding to $\rho \approx 1.0, 0.857, 0.714, 0.571$. Crucially, $\rho$ does not change the network's weights at all — it changes the spatial size $H \times W$ of every intermediate feature map.

Derive its effect. Every convolution cost has an explicit $H \cdot W$ factor. Scaling input resolution by $\rho$ scales every feature map's height and width by $\rho$, so $H W \to (\rho H)(\rho W) = \rho^2 H W$. Therefore

$$
\text{FLOPs}(\rho) = \rho^2 \cdot \text{FLOPs}(1.0).
$$

Compute scales as $\rho^2$ — quadratically, again — but notice the difference from $\alpha$: **$\rho$ reduces compute without reducing parameters**, because it only touches activation spatial sizes, not weight tensors. That distinction matters enormously on memory-constrained devices, a point we will return to. Resolution at 160 instead of 224 gives roughly $(160/224)^2 \approx 0.51$ of the compute at the same parameter count.

### One base model, a whole family

Put the two dials together and the total compute scales as the product:

$$
\text{FLOPs}(\alpha, \rho) \approx \alpha^2 \rho^2 \cdot \text{FLOPs}(1.0, 1.0).
$$

This is the conceptual mental model that made V1 matter: you do not ship a model, you ship a *family*, and you place each member on an accuracy-versus-compute curve by choosing $(\alpha, \rho)$. A wearable with a 50-MFLOP budget picks a small $\alpha$ and small input; a flagship phone with headroom picks $\alpha = 1.0$ at 224. Same architecture, same training recipe, one curve.

![Matrix showing how width multiplier alpha and resolution multiplier rho trade MobileNet FLOPs and parameters against top-1 accuracy](/imgs/blogs/the-mobilenet-family-2.png)

The matrix above lays out the trade concretely. Read the rows as members of the same family. The full model ($\alpha=1.0$, res 224) costs 569 MFLOPs for 70.6%. Thinning to $\alpha=0.5$ cuts compute to 149 MFLOPs (the $\alpha^2$ effect, roughly a 4x drop) and accuracy to 63.7%. Dropping resolution to 160 at full width cuts to 290 MFLOPs (the $\rho^2$ effect) but keeps all 4.2M parameters — useful when you are compute-bound but not memory-bound. The bottom row stacks both knobs: $\alpha=0.5$ at res 128 reaches a tiny 49 MFLOPs, small enough for serious embedded targets, at 56.3% top-1. That is the dial in action.

#### Worked example: hitting a wearable's compute budget

Suppose you are shipping a gesture classifier onto a smartwatch SoC that can sustain about 60 million multiply-adds per inference if you want 15 frames per second within the thermal envelope. The full MobileNetV1 needs 569 MFLOPs — roughly 9.5x over budget. You need to cut compute by a factor of about 9.5.

Using $\text{FLOPs} \approx \alpha^2 \rho^2 \cdot 569$, you need $\alpha^2 \rho^2 \approx 1/9.5 \approx 0.105$. One clean solution: $\alpha = 0.5$ (factor $0.25$) and $\rho = 128/224 \approx 0.571$ (factor $0.327$). Product: $0.25 \times 0.327 = 0.082$, giving $0.082 \times 569 \approx 47$ MFLOPs — comfortably under 60. From the family table, $\alpha=0.5$ at res 128 lands around 56% top-1. If that accuracy is too low for your gesture set, you have a clear lever to trade back: bump resolution to 160 ($\rho^2 = 0.51$) for $0.25 \times 0.51 \times 569 \approx 73$ MFLOPs (slightly over budget, so maybe drop to 12 FPS), buying a few points of accuracy. The point is that the math turns a vague "make it smaller" into a small set of concrete, predictable options.

The honest caveat: FLOPs are not latency. We will hammer this repeatedly because it is the central lesson the family teaches. A depthwise convolution has terrible *arithmetic intensity* — it does very little compute per byte of memory it touches — so it is often memory-bound rather than compute-bound. Two models with the same FLOPs can have very different latency depending on how memory-bound their operations are. The [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) is the right tool for reasoning about this, and it is precisely why V3 stopped optimizing FLOPs and started optimizing measured latency directly.

## MobileNetV2: inverted residuals and linear bottlenecks

V1 was a stack of blocks with no skip connections, and that limited how deep you could usefully make it — gradients have a hard time flowing through a long chain of depthwise-separable layers, and the representational story is awkward. MobileNetV2 (Sandler et al., 2018) redesigned the block itself. The new block has two ideas in its name: the **inverted residual** and the **linear bottleneck**. Both are counterintuitive, and both are right for the specific physics of depthwise convolution.

Start with the standard residual block you know from ResNet. It follows a *wide-narrow-wide* pattern: take a wide feature map, squeeze it down to a narrow bottleneck with a $1\times1$ convolution (cheap), do the expensive $3\times3$ convolution in the narrow space, then expand back to wide with another $1\times1$. The skip connection runs over the wide tensors. This makes sense for standard convolutions, where the $3\times3$ is so expensive that you want to run it on as few channels as possible.

The inverted residual flips that. It follows a *narrow-wide-narrow* pattern: take a narrow bottleneck, **expand** it to a wide space with a $1\times1$ convolution, do the $3\times3$ **depthwise** convolution in the wide space, then **project** back down to narrow with another $1\times1$. The skip connection runs over the *narrow* bottleneck tensors. Why invert it? Because the $3\times3$ here is depthwise, and depthwise is cheap — it scales linearly in channels, not quadratically. So you can afford to run it on a *wide* feature map, which gives the per-channel filters more channels to work with and produces a richer representation. The expensive part is now the $1\times1$ convolutions, and you keep those connected to the narrow ends.

### The expansion factor and why depthwise wants room

The expansion factor $t$ controls how much the $1\times1$ expand widens the bottleneck. V2 uses $t=6$ almost everywhere: a bottleneck of $C$ channels expands to $6C$, the depthwise filters operate on $6C$ channels, then the projection brings it back down. The intuition is that a depthwise convolution is a weak operator — each output channel only sees one input channel's spatial neighborhood, with no cross-channel mixing inside the depthwise step. Giving it a wide $6C$ space means many of those weak per-channel filters can specialize and collectively capture more, before the $1\times1$ projection mixes them back together. Run depthwise on a thin tensor and it is starved; run it on a fat tensor and it has room.

There is a real cost to this. The wide $6C$ activation tensor in the middle of the block is the largest intermediate the block produces. If you materialize it fully, the inverted residual is *more* memory-hungry at peak than you might expect from its narrow input and output. V2's authors were acutely aware of this and designed around it, which brings us to the memory story.

Let us count the compute of one inverted-residual block exactly, because the numbers explain where it spends. With input bottleneck $C$ channels, expansion factor $t$, output bottleneck $C'$ channels, spatial size $H \times W$, and $3\times3$ depthwise, the three sub-operations cost:

$$
\underbrace{H W \cdot C \cdot tC}_{\text{1x1 expand}} \;+\; \underbrace{H W \cdot tC \cdot 9}_{\text{3x3 depthwise}} \;+\; \underbrace{H W \cdot tC \cdot C'}_{\text{1x1 project}}.
$$

The two $1\times1$ convolutions (expand and project) dominate, each scaling with the product of a bottleneck width and the expanded width $tC$; the depthwise term carries only a factor of 9, tiny by comparison. So the expansion factor $t$ shows up *linearly* in every term — double $t$ and you roughly double the block's compute. That is the dial V2 uses to trade capacity for cost block by block, and it is why V2 keeps $t=6$ in the middle of the network but drops it (sometimes to $t=1$, a degenerate "no expansion" block) in the first block where the input is already thin. The table below contrasts the two block philosophies.

| Property | ResNet residual (wide-narrow-wide) | V2 inverted residual (narrow-wide-narrow) |
|---|---|---|
| Skip connection runs over | wide tensors | thin bottleneck tensors |
| Expensive $3\times3$ operates on | narrow space | wide expanded space |
| $3\times3$ type | standard (quadratic in channels) | depthwise (linear in channels) |
| Final projection activation | ReLU | none (linear bottleneck) |
| Peak inference activation memory | higher (wide on skip path) | lower (thin on skip path) |

![Vertical stack showing the MobileNetV2 inverted residual block from thin bottleneck through expand depthwise and linear projection to the residual add](/imgs/blogs/the-mobilenet-family-3.png)

The stack above walks the block top to bottom. Note the order: the residual add at the bottom connects the *bottleneck* tensors (thin), not the expanded ones. That is the inverted part. And the projection step is labeled linear — no activation — which is the second idea.

### The linear bottleneck and why ReLU on a thin tensor destroys information

The "linear" in linear bottleneck means the final $1\times1$ projection has **no nonlinearity** after it. Most blocks end with a ReLU; V2's inverted residual deliberately does not, on the narrow projection. The reasoning is subtle and worth doing carefully because it is one of the more elegant arguments in efficient-architecture design.

ReLU zeroes out all negative values. When you apply ReLU to a high-dimensional tensor, the information lost (the negative parts) can often be recovered by the rest of the network, because the representation is redundant — there are many channels carrying overlapping information. But when you apply ReLU to a *low-dimensional* tensor — a thin bottleneck — there is little redundancy, and zeroing out the negatives can collapse the manifold the data lives on irreversibly. Sandler et al. demonstrate this with a toy experiment: embed a spiral into $n$ dimensions, apply ReLU, project back, and watch the reconstruction. At low $n$ the spiral is mangled; at high $n$ it survives. The conclusion: the wide expanded space can tolerate ReLU (it does use ReLU6 there), but the narrow bottleneck cannot, so the projection into the bottleneck must be **linear** to preserve the information the residual connection carries.

This is not a minor tweak. The V2 paper shows that adding a nonlinearity to the bottleneck *hurts* accuracy by a measurable margin. The linear bottleneck is doing real work: it keeps the low-dimensional residual highway clean so information flows across many blocks without being repeatedly clipped.

### Memory-efficient inference

Here is the practical payoff that makes V2 a darling of edge deployment. Because the block's input and output are both thin bottlenecks and only the *intermediate* expanded tensor is fat, a smart runtime never needs to hold the full expanded tensor in memory at once. The expand-depthwise-project pipeline can be computed in *channel groups* (or spatial tiles): expand a slice of the bottleneck, filter it, project it, accumulate into the output, then move to the next slice. The peak memory the block needs is bounded by the bottleneck tensors plus one small working slice of the expansion — not the whole $6C$ tensor.

Concretely, V2's authors show that peak memory for inference can be bounded by roughly

$$
M \approx \max_{\text{blocks}} \left( |\text{input bottleneck}| + |\text{output bottleneck}| + \text{small working set} \right),
$$

which is dominated by the thin tensors, not the fat intermediate. On a device where SRAM or the activation buffer is the binding constraint — and on phones and microcontrollers it very often is — this is the difference between fitting and not fitting. A standard wide-narrow-wide residual block, by contrast, keeps wide tensors on the skip path, so its peak activation memory is higher. V2 gets you both lower peak memory *and* higher accuracy than V1 at comparable FLOPs (about 72.0% top-1 at roughly 300 MFLOPs and 3.5M parameters, versus V1's 70.6% at 569 MFLOPs). That is moving the Pareto frontier, not sliding along it.

#### Worked example: peak activation memory, V2 versus a plain residual

Take a block midway through the network operating on a $14\times14$ feature map with a 64-channel bottleneck and expansion factor $t=6$, using int8 activations (1 byte each).

For V2's inverted residual, the fat intermediate is $14 \times 14 \times (6 \times 64) = 14 \times 14 \times 384 = 75{,}264$ bytes if fully materialized. But the input and output bottlenecks are each only $14 \times 14 \times 64 = 12{,}544$ bytes. A tiling runtime holds the two bottlenecks (about 25 KB combined) plus a working slice of the expansion — say one-sixth of it, about 12.5 KB — for a peak near **38 KB**, not 75 KB.

A wide-narrow-wide block targeting the same expressive width would keep a wide ~384-channel tensor live across its skip path: peak nearer **75 KB or more**. On a Cortex-M7 with 320 KB of total SRAM shared across the whole inference, a 2x difference in per-block peak is the difference between the model fitting with room for the framebuffer and the model not fitting at all. This is why, when people put real classifiers on microcontrollers, the inverted-residual shape keeps showing up — see the [edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for what those SRAM budgets actually look like.

## MobileNetV3: let the search do the designing

By 2019 a new idea had matured: instead of a human picking block types, channel counts, kernel sizes, and where to downsample, let a search algorithm explore that space — but reward it for *measured latency on the actual target device*, not for FLOPs. This is platform-aware neural architecture search, pioneered by MnasNet (Tan et al., 2019), and MobileNetV3 (Howard et al., 2019) is the result of pointing that machinery at the mobile-classifier problem and then hand-finishing the result. V3 is not one new idea; it is a *process*, plus a handful of cheap, well-chosen components.

![Graph of the MobileNetV3 design pipeline branching from a latency target through platform-aware NAS and NetAdapt and head redesign into the final network](/imgs/blogs/the-mobilenet-family-5.png)

The pipeline above shows the shape of it. You start with a latency *target* — a millisecond budget on a named phone. A platform-aware NAS (MnasNet-style) searches a space of block configurations and, critically, includes *real measured latency on the device* in its reward, so it learns to prefer operations that are actually fast on that silicon rather than ones that merely have low FLOPs. That produces a coarse block layout. NetAdapt then fine-tunes it: starting from the NAS network, it greedily trims the number of filters in individual layers, one layer at a time, re-measuring latency and accuracy after each trim, keeping the trims that best preserve accuracy per unit latency saved. Finally — and this is the part people forget — the authors hand-redesigned the expensive head and tail that the search had left bloated. Two refinement paths (NetAdapt and the manual head/tail surgery) merge into the final V3-Large and V3-Small networks. We will go through the components the search and the humans converged on.

### Platform-aware NAS optimizes the right objective

The deep reason V3 beats hand-designed V2 at the same latency is that it optimizes the *right objective*. MnasNet-style search uses a reward roughly of the form

$$
\text{maximize} \quad \text{Accuracy}(m) \times \left[ \frac{\text{Latency}(m)}{T} \right]^{w}
$$

where $\text{Latency}(m)$ is the *measured* on-device latency of candidate model $m$, $T$ is the target latency, and $w$ is a negative weight that penalizes models slower than $T$ and rewards faster ones. Because latency is measured on the device, the search naturally avoids operations that look cheap in FLOPs but run slowly on the hardware — and embraces operations that the hardware happens to accelerate. A human designer reasoning about FLOPs simply cannot internalize the device's quirks (how its caches behave, which kernels its libraries have optimized, where memory bandwidth bites) the way a search loop with a real latency measurement in the reward can. That is the NAS gain, and it is measurable: V3-Large reaches 75.2% top-1 at about 51 ms on a Pixel-1, versus V2's 72.0% at about 75 ms. Three points more accurate *and* faster. This is the same idea generalized in [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas), which the dedicated post in this series develops further.

The search space MnasNet and V3 explore is not unconstrained — that would be hopeless. It is *factorized*: the network is divided into a fixed number of blocks, and the search picks, per block, the convolution kernel size (3 or 5), the expansion factor, the output channel count, whether to include a squeeze-and-excite module, and which activation to use. Keeping the space factorized and block-structured is what makes the search tractable and what keeps the result a recognizable MobileNet rather than some unbuildable tangle. The reward in the equation above is what the controller (a reinforcement-learning policy in MnasNet) maximizes by sampling architectures, training them briefly, measuring their latency on the phone, and updating toward the high-reward region. The expensive part is the on-device latency measurement, which is why a latency *predictor* — a lookup table of measured per-operation costs — is often used to approximate it cheaply during the inner loop.

### NetAdapt: trimming layer by layer

NAS produces a coarse architecture; NetAdapt (Yang et al., 2018) refines it. The procedure is a greedy, latency-budgeted shrink. Starting from the NAS network, at each iteration NetAdapt proposes, for *every* layer independently, a candidate that removes some filters from that layer until the whole network's latency drops by a small fixed step (say, a 1% latency reduction per iteration). Each candidate is briefly fine-tuned and its accuracy measured. Among all the per-layer candidates that hit the latency-reduction target, NetAdapt keeps the *one* that retains the most accuracy — the layer that could give up filters most cheaply — and discards the rest. Then it repeats from the new, slightly smaller network, ratcheting the latency down step by step until it reaches the target, with a final longer fine-tune at the end.

The elegance is that NetAdapt never needs an explicit "importance" heuristic for which filters to cut — it lets the measured accuracy-after-fine-tuning decide, layer by layer, where the slack is. It is structured pruning guided by a direct latency measurement, which is exactly the right tool for the edge and which connects to the [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up) post in this series. The combination — NAS for the coarse structure, NetAdapt for the fine filter counts — is what lets V3 land precisely on a latency target rather than near it.

### h-swish: a cheap activation, derived

The swish activation, $\text{swish}(x) = x \cdot \sigma(x)$ where $\sigma$ is the sigmoid, consistently improves accuracy over ReLU in deeper networks. But it is expensive on edge hardware: the sigmoid requires an exponential, $\sigma(x) = 1/(1 + e^{-x})$, and computing $e^{-x}$ per element is slow on mobile CPUs and outright painful in int8, where there is no native cheap exponential. So V3 introduces **hard-swish** (h-swish), which approximates swish using only operations the hardware already loves.

The construction starts with a hard approximation to the sigmoid. The *hard sigmoid* replaces the smooth $\sigma$ with a clamped linear ramp:

$$
\text{h-sigmoid}(x) = \frac{\text{ReLU6}(x + 3)}{6} = \frac{\min(\max(x + 3, 0), 6)}{6}.
$$

Read it left to right: shift $x$ up by 3, clamp to the range $[0, 6]$ with ReLU6, and divide by 6. The result is 0 for $x \le -3$, 1 for $x \ge 3$, and a straight line in between — a piecewise-linear stand-in for the S-curve. Now define h-swish by substituting it into swish's shape:

$$
\text{h-swish}(x) = x \cdot \text{h-sigmoid}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}.
$$

Count the operations: an add (x + 3), a clamp (ReLU6, which is two comparisons), a multiply (by x), and a multiply or shift (by 1/6). No exponential, no division that cannot be folded into a constant. Every one of these is cheap in fp32 *and* cheap in int8, and ReLU6's bounded output range $[0, 6]$ is friendly to quantization because the range is known and fixed. The accuracy is essentially identical to swish — the curves track each other closely — but the cost is a handful of integer-friendly operations.

![Before-and-after comparison of swish requiring an exponential versus hard-swish using only clamp and multiply operations](/imgs/blogs/the-mobilenet-family-4.png)

The figure contrasts the two. The left column is swish: smooth, accurate, but gated by an `exp()` that integer hardware cannot do cheaply. The right column is h-swish: no exponential, near-identical accuracy, and quantization-friendly because of the bounded ReLU6 range. V3 does not even use h-swish everywhere — it uses ReLU in the early layers (where activations are large and h-swish's overhead matters more, and where ReLU is good enough) and switches to h-swish only in the deeper, lower-resolution layers where the accuracy benefit is worth the cost. That selective placement is itself a latency-driven decision.

#### Worked example: the cost of exp() versus h-swish on a feature map

Consider a single V3 activation applied to a $14 \times 14 \times 112$ feature map — 21,952 elements.

With swish, each element needs an exponential. A reasonable software `expf()` on a mobile CPU without a hardware transcendental unit costs on the order of 20 to 40 floating-point operations equivalent (polynomial approximation, range reduction). Call it 30. So swish costs about $21{,}952 \times (30 + 2) \approx 700{,}000$ operation-equivalents for this one activation, and in int8 you would have to dequantize, do the exp in float, and requantize — even worse.

With h-swish, each element needs roughly 4 cheap operations (add, two-sided clamp counted as two, multiply, and a fold-in scale). Call it 5. So h-swish costs about $21{,}952 \times 5 \approx 110{,}000$ operation-equivalents, all integer-friendly. That is a ~6x reduction *per activation*, and a network has many activations. Spread across the whole V3-Large, the authors report h-swish (with the optimized implementation) adds negligible latency while recovering most of swish's accuracy benefit — which would have been unaffordable with real swish on the same device.

### Squeeze-and-excite and the redesigned head and tail

V3 adds two more components the search found valuable. **Squeeze-and-excite (SE)** blocks add a tiny channel-attention mechanism: global-average-pool the feature map to one value per channel, pass through a small two-layer bottleneck (with h-swish and a hard-sigmoid gate), and use the result to rescale each channel. It costs very little compute — the pooled vector is small — but lets the network emphasize informative channels and suppress others, buying accuracy cheaply. V3 places SE blocks inside the inverted residuals where the search said they pay off, and uses the hard-sigmoid (not the smooth one) for the gate, again to avoid the exponential.

Why does SE cost so little? The squeeze pools an $H \times W \times C$ feature map down to a $1 \times 1 \times C$ vector, so the two fully connected layers that follow operate on $C$ numbers, not $H W C$. For a typical block with $C$ in the low hundreds and a reduction ratio of 4, the SE module adds on the order of $C^2/4$ multiply-adds — a few tens of thousands — against a block that costs millions. It is, in compute terms, almost free, and the accuracy it buys (typically 0.5 to 1 point across the network) is a bargain. The one place SE bites is *latency on some NPUs*: the global pool and the per-channel rescale are awkward shapes for accelerators tuned for dense convolutions, and if the delegate does not support them they fall back to CPU. This is exactly the kind of trade-off only a measured-latency search can navigate, which is why V3 has SE in some blocks and not others rather than uniformly — the search weighed the accuracy gain against the per-chip latency cost block by block.

The **redesigned head and tail** is the unglamorous but high-impact hand surgery. The NAS-found network had an expensive final stage: V2's last block expanded to a wide tensor before the global pool, which is wasteful because after pooling the spatial dimension is gone. V3 moves the final $1\times1$ expansion to *after* the average pool, where it operates on a $1\times1$ spatial map and is nearly free, and removes a now-redundant projection layer. This single change saves about 7 ms on the Pixel with no accuracy loss — roughly 11% of the V3-Large latency for free. Similarly, V3 reduces the initial convolution from 32 filters to 16 (the search found 16 was enough with h-swish), saving another couple of milliseconds. These are the kinds of optimizations a human finds by *staring at the latency profile* of a search-produced network, and they are why V3 is "NAS plus finishing," not pure NAS.

### V3-Large versus V3-Small

V3 ships in two sizes, found by running the search with two different latency targets. **V3-Large** targets higher-accuracy use cases: about 5.4M parameters, 219 MFLOPs, 75.2% top-1, around 51 ms on Pixel-1. **V3-Small** targets the tightest budgets: about 2.5M parameters, 56 MFLOPs, 67.4% top-1, around 16 ms on Pixel-1. They are not the same network scaled by a multiplier — they are *separately searched* networks with different block layouts, because the optimal architecture genuinely differs at different budgets. You can still apply a width multiplier on top of either (V3-Small at $\alpha = 0.75$, say) to fine-tune further, but the two base networks already span most of the useful range.

The fact that the two sizes have *different layouts*, not just different channel counts, is the deepest lesson of V3. When you have very little compute, the search learns to spend it differently — fewer SE blocks, different kernel sizes, a leaner head — than when you have more. A single architecture scaled by a multiplier is forced to keep the same proportions at every size, which is suboptimal at the extremes. This observation is exactly what EfficientNet would systematize a year later with compound scaling: rather than scaling one dimension, scale width, depth, and resolution together by a balanced ratio, because the right proportions shift as the budget grows. V3's two separately-searched sizes are an early, manual version of the same insight, and they are why "MobileNetV3" is really a small family rather than a single net.

## The family in one table

Let us put all three generations side by side. The numbers below are from the respective papers and the standard `torchvision` reference implementations, at $224 \times 224$ input and $\alpha = 1.0$ unless noted, measured top-1 on ImageNet. Latency figures are batch-1 on a Pixel-1 CPU as reported in the V3 paper; treat them as relative, not absolute, because they depend heavily on the runtime and thermal state.

| Model | Params (M) | MFLOPs | ImageNet top-1 | Pixel-1 latency (ms) |
|---|---|---|---|---|
| MobileNetV1 1.0 | 4.2 | 569 | 70.6% | ~113 |
| MobileNetV2 1.0 | 3.5 | 300 | 72.0% | ~75 |
| MobileNetV3-Small | 2.5 | 56 | 67.4% | ~16 |
| MobileNetV3-Large | 5.4 | 219 | 75.2% | ~51 |

Read the table as the story of the family. V1 to V2: roughly half the FLOPs (569 to 300), fewer parameters (4.2 to 3.5M), and *higher* accuracy (70.6% to 72.0%) — the inverted residual and linear bottleneck simply being a better block. V2 to V3-Large: again fewer FLOPs (300 to 219), about the same parameters, and a further 3.2-point jump in accuracy (72.0% to 75.2%) plus lower latency — the NAS gain plus h-swish and SE. And V3-Small shows what the family can do at the bottom of the budget: 67.4% top-1 at just 56 MFLOPs and ~16 ms, which is genuinely usable on extremely constrained hardware.

![Matrix comparing MobileNet V1, V2, V3-Small, and V3-Large on parameters, FLOPs, top-1 accuracy, and Pixel-1 latency](/imgs/blogs/the-mobilenet-family-6.png)

The matrix above is the same comparison rendered so the wins jump out. Scan the FLOPs column top to bottom and watch it fall while top-1 holds or rises — that monotone improvement, generation over generation, is the whole point of the family. Each new design did not just make a smaller model; it made a *better* model that was also smaller. That is the signature of a moving Pareto frontier.

![Before-and-after comparison of MobileNetV2 and MobileNetV3-Large at a matched mobile latency budget showing the NAS accuracy gain](/imgs/blogs/the-mobilenet-family-7.png)

The before-after figure isolates the cleanest comparison: V2 versus V3-Large. The hand-designed V2 needs about 75 ms for 72.0%; the NAS-found V3-Large delivers 75.2% at about 51 ms. Faster *and* more accurate. There is no width-multiplier trick that gets V2 to V3-Large's point — you cannot slide along V2's curve to reach it, because it sits on a *different, better* curve. That is the dividend of optimizing measured latency with search instead of optimizing FLOPs by hand.

## Practical: load, scale, and quantize MobileNet

Enough theory. Here is how you actually use this family. Everything below runs against `torchvision` and PyTorch's quantization stack; `timm` exposes a wider zoo (including the V3 variants and many width multipliers) if you want more options.

### Loading the models

`torchvision` ships V2 and both V3 variants with pretrained ImageNet weights. Loading them is one line each.

```python
import torch
import torchvision.models as models

# MobileNetV2 (width multiplier 1.0)
v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
v2.eval()

# MobileNetV3-Large and V3-Small
v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
v3_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
v3_large.eval()
v3_small.eval()

# Sanity check: parameter counts
for name, m in [("v2", v2), ("v3_large", v3_large), ("v3_small", v3_small)]:
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{name}: {n:.2f}M params")
# v2: 3.50M params
# v3_large: 5.48M params
# v3_small: 2.54M params
```

### Applying a width multiplier

`torchvision`'s `mobilenet_v2` accepts a `width_mult` argument directly, so you can construct a thinned model (you would then train it, since pretrained weights only exist for the default widths). `timm` exposes pre-trained thinned variants by name.

```python
import torchvision.models as models
import timm

# A 0.5-width MobileNetV2 from torchvision (random init -- you must train it)
v2_half = models.mobilenet_v2(width_mult=0.5)
n_full = sum(p.numel() for p in models.mobilenet_v2().parameters())
n_half = sum(p.numel() for p in v2_half.parameters())
print(f"width 0.5 -> {n_half / n_full:.2f}x params")  # ~0.30x (not exactly 0.25
# because the classifier head and a few fixed layers do not scale)

# Pretrained thinned / scaled variants via timm
m = timm.create_model("mobilenetv3_small_050", pretrained=True)  # alpha=0.5 V3-Small
m = timm.create_model("mobilenetv2_140", pretrained=True)         # alpha=1.4 V2
```

Notice the ratio is about 0.30x, not the clean 0.25x the $\alpha^2$ law predicts. The law applies to the convolutional body; the classifier head and a few non-scaled layers dilute it. Always *measure* the actual parameter and FLOP count of your scaled model rather than trusting the idealized formula — the formula is for reasoning, the profiler is for truth.

### Quantizing to int8: the canonical example

MobileNet is the textbook case for post-training quantization because its architecture is, by design, quantization-friendly: bounded ReLU6 and hard-sigmoid activations keep ranges tight and predictable, and there are no exotic operations that lack int8 kernels. The full reasoning behind calibration and per-channel scales lives in [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq); here is the runnable flow with the modern `torch.ao.quantization` FX API.

```python
import torch
import torchvision.models as models
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import copy

model = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
).eval()

# x86 server-side int8 backend; use "qnnpack" for ARM mobile
qconfig_mapping = get_default_qconfig_mapping("x86")
example_inputs = (torch.randn(1, 3, 224, 224),)

# Insert observers (records activation ranges during calibration)
prepared = prepare_fx(copy.deepcopy(model), qconfig_mapping, example_inputs)

# Calibrate: run a few hundred representative images through the observers.
# No labels, no backprop -- just forward passes to estimate ranges.
def calibrate(prepared_model, data_loader, n_batches=32):
    prepared_model.eval()
    with torch.inference_mode():
        for i, (images, _) in enumerate(data_loader):
            prepared_model(images)
            if i + 1 >= n_batches:
                break

calibrate(prepared, calib_loader)  # calib_loader yields representative images

# Convert observers + fake-quant into real int8 ops
int8_model = convert_fx(prepared)

# Size comparison
import os
for tag, m in [("fp32", model), ("int8", int8_model)]:
    torch.save(m.state_dict(), f"/tmp/mnv3_{tag}.pt")
    mb = os.path.getsize(f"/tmp/mnv3_{tag}.pt") / 1e6
    print(f"{tag}: {mb:.2f} MB")
# fp32: ~21.9 MB
# int8: ~5.6 MB   (~3.9x smaller)
```

The whole flow is: pick a backend (`qnnpack` for ARM, `x86` for Intel/AMD), insert observers with `prepare_fx`, run a few hundred representative images to calibrate activation ranges (no labels, no training), then `convert_fx` to bake the int8 operations. For MobileNetV3-Large the int8 model is about 5.6 MB versus 21.9 MB in fp32 — roughly 3.9x smaller — and on a CPU with good int8 kernels it runs about 2 to 4x faster. If post-training quantization drops accuracy more than you can spend (V3's SE blocks and the head can be slightly sensitive), the next step is quantization-aware training, but for MobileNet, PTQ usually gets you within a point.

### Benchmarking honestly on CPU

Benchmarks lie when you do them carelessly. You must warm up (the first few inferences pay for kernel selection, page faults, and JIT), you must measure many iterations, you must report a percentile not just the mean, and you must pin the batch to 1 because that is the on-device reality. Here is a benchmark harness that does it right.

```python
import torch, time, numpy as np

def benchmark(model, input_shape=(1, 3, 224, 224), warmup=20, iters=200, threads=4):
    torch.set_num_threads(threads)  # match your target's core count
    model.eval()
    x = torch.randn(*input_shape)
    with torch.inference_mode():
        for _ in range(warmup):       # warm up: discard these
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
    times = np.array(times)
    print(f"p50={np.percentile(times,50):.1f} ms  "
          f"p99={np.percentile(times,99):.1f} ms  "
          f"mean={times.mean():.1f} ms")
    return times

# Example (numbers are illustrative; yours depend on the exact CPU):
# fp32 V3-Large: p50=24.3 ms  p99=31.7 ms
# int8 V3-Large: p50= 9.1 ms  p99=12.4 ms   (~2.7x faster)
```

The gap between p50 and p99 is the tail you will be paged about. On a thermally throttled phone the p99 can be several times the p50 once the device heats up, which is why a 200-iteration loop on a cool desk underestimates field latency. When you graduate from a CPU microbenchmark to the real device, measure on the actual runtime — TensorFlow Lite with the NNAPI or Core ML delegate, ONNX Runtime with the NNAPI execution provider — because the int8 kernels and the NPU offload there are what determine production latency, and they differ sharply from PyTorch CPU. The series' [metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) post goes deep on measuring this correctly.

### Deploying to the real edge runtimes

PyTorch CPU is a fine place to prototype, but you ship through a mobile runtime. The two dominant paths are TensorFlow Lite (now LiteRT) for Android and Core ML for iOS, with ONNX Runtime as the cross-platform middle ground. MobileNet is the smoothest possible model to push through any of them, because every runtime has hand-tuned int8 depthwise and $1\times1$ kernels for exactly this architecture. Here is the TFLite path, the most common for Android, going straight from a Keras MobileNet to a full-int8 `.tflite` with a representative calibration set.

```python
import tensorflow as tf
import numpy as np

# Keras ships MobileNetV2/V3 directly
keras_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3), weights="imagenet", include_top=True
)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]      # enable quantization

# Representative dataset: a generator yielding a few hundred real inputs.
# This is the calibration set -- it sets activation ranges. No labels needed.
def representative_data_gen():
    for _ in range(200):
        # use REAL preprocessed images in production, not random noise
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_data_gen
# Force full int8 (weights AND activations) so it runs on int8-only NPUs:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
with open("mobilenet_v3_large_int8.tflite", "wb") as f:
    f.write(tflite_int8)
print(f"int8 tflite: {len(tflite_int8) / 1e6:.2f} MB")  # ~6 MB
```

Two flags carry all the weight. `Optimize.DEFAULT` turns on quantization; the `representative_dataset` is your calibration loop (the same role the PyTorch observers played). Forcing `TFLITE_BUILTINS_INT8` and int8 input/output types produces a model that runs entirely in integer arithmetic, which is what an int8-only NPU like the one in many phone SoCs requires — if you leave operations in float, the runtime falls back to CPU for those ops and the NPU offload is wasted. Once you have the `.tflite`, you benchmark it on the device with the official `benchmark_model` tool, which reports per-op timing and tells you exactly which operations the delegate accelerated and which fell back.

```bash
# Run on-device (adb push the binary + model to the phone first).
# --use_nnapi routes int8 ops to the NPU; check the delegate summary.
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v3_large_int8.tflite \
  --num_threads=1 \
  --use_nnapi=true \
  --enable_op_profiling=true \
  --warmup_runs=20 \
  --num_runs=200
```

The op profile is where MobileNet deployment goes right or wrong. If h-swish or a squeeze-and-excite op is not supported by the NNAPI delegate on a given chip, that op falls back to CPU, the tensor bounces off the NPU and back, and the round trip can cost more than the op saved. This is the single most common MobileNet-on-device surprise: a model that should fly stutters because one unsupported op forces a CPU-NPU ping-pong. The fix is to check the delegate's supported-op list for your target and, if needed, swap the offending activation (h-swish for ReLU in that layer) to keep the whole graph on the accelerator. The [edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle) post treats this fallback debugging as a first-class workflow.

### The int8 before-and-after, measured

Here is the headline result the whole quantization story promises, for MobileNetV3-Large, stated honestly with the caveat that exact numbers depend on the chip and runtime. The fp32 figures are the PyTorch CPU baseline; the int8 figures are the full-integer TFLite model on a mid-range Android SoC's NPU; treat the absolute latencies as representative, not gospel.

| Metric | fp32 baseline | int8 (full integer) | Delta |
|---|---|---|---|
| Model size on disk | ~21.9 MB | ~5.6 MB | 3.9x smaller |
| ImageNet top-1 | 75.2% | ~74.4% | -0.8 pts |
| Latency (NPU, batch 1) | n/a (CPU only) | ~7 ms | runs on NPU |
| Latency (CPU, batch 1) | ~24 ms | ~9 ms | ~2.7x faster |
| Peak activation memory | ~1.0x | ~0.5x | int8 activations halve it |

Read this as the canonical MobileNet quantization win: a tenth of a percent under one point of accuracy lost, four times smaller on disk, roughly three times faster on CPU, and — the real prize — small and integer-only enough to run on the NPU at all, which a float model cannot. This is why MobileNet is the example every quantization tutorial reaches for. The architecture was built with bounded, quantization-friendly activations, so post-training quantization, which is the cheapest lever in the whole [taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), gets you almost all the way with a few hundred unlabeled calibration images and no retraining.

#### Worked example: a doorbell camera on a \$15 NPU module

Suppose you are building a smart doorbell that classifies "person / package / animal / nothing" on a tiny \$15 module with an int8-only NPU rated at about 1 TOPS and 8 MB of usable model storage, and the product needs to react within 100 ms of motion so it can start recording.

Start from MobileNetV3-Large: 219 MFLOPs, but it must run in int8 (the NPU has no float path) and fit in 8 MB. The fp32 model is 21.9 MB — too big — and float-only, so it would not run on the NPU at all. After full-int8 TFLite conversion it is 5.6 MB (fits, with 2.4 MB to spare for the runtime) and runs entirely on the NPU at roughly 7 ms per inference. Even allowing 20 ms for camera capture and preprocessing and a generous 10 ms of NPU-CPU overhead, you are at ~37 ms — comfortably inside the 100 ms budget, with thermal and battery headroom to run continuously. If you needed even more margin, V3-Small (56 MFLOPs, ~16 ms CPU / a few ms NPU, 2.5 MB int8) buys it at the cost of dropping to ~67% top-1, which for a four-class doorbell problem fine-tuned on your own data is almost certainly fine. The decision is not "which is most accurate" but "which fits the int8-only, 8 MB, 100 ms envelope" — and that is the edge engineer's mindset in one sentence.

### Problem-solving and stress tests

Let us reason through the failure modes, because that is where the real engineering lives.

**What happens when you quantize the depthwise layers per-tensor?** This is the classic MobileNet quantization trap. A depthwise layer has one filter per channel, and different channels can have wildly different weight magnitudes — one channel's filter might span $[-0.02, 0.02]$ while another spans $[-2, 2]$. A single per-tensor scale must cover the widest channel, which crushes the small-magnitude channels into a handful of quantization levels and destroys their information. The symptom is a MobileNet that loses 5 or more points of top-1 from int8, far more than the under-one-point you should see. The fix is *per-channel* weight quantization (a separate scale per output channel), which is nearly free and is the default in every serious quantization toolkit precisely because of MobileNet. If your int8 MobileNet is hemorrhaging accuracy, check this first.

**What happens at int4?** The bounded activations help, but depthwise-separable networks are already lean — they have removed the redundancy that absorbs quantization noise. Pushing MobileNet to 4-bit weights typically costs several points of accuracy with plain post-training quantization, and you need quantization-aware training to recover. The lesson generalizes: the leaner the architecture, the less quantization headroom it has, because there is less redundancy to spend. A fat over-parameterized network quantizes more gracefully than a tight efficient one — an irony worth remembering when you stack the efficiency levers.

**What happens when the calibration set is tiny or unrepresentative?** Calibration estimates activation ranges from the data you show it. Feed it 16 images from one class and the ranges will be wrong for the rest, clipping activations the model relies on. A few hundred images spanning the class distribution is the sweet spot; more than a thousand rarely helps. Random noise (as in the toy generator above) is fine for a smoke test but wrong for production — use real preprocessed inputs.

**What happens when the model is memory-bound, not compute-bound?** Then shaving FLOPs with a smaller $\alpha$ buys less latency than you expect, because the depthwise steps that dominate the wall clock are gated by memory bandwidth, not arithmetic. On such a target, dropping *resolution* (which shrinks the activation tensors that the memory system moves) can buy more latency per accuracy point than dropping *width*. This is a concrete, actionable consequence of the arithmetic-intensity argument: when memory-bound, prefer $\rho$ over $\alpha$. Profile to find out which regime you are in before you pick a knob.

## Case studies and real numbers

A few concrete results from the literature and from shipped systems, to anchor the claims in measured reality.

**MobileNetV3-Large on Pixel phones.** The V3 paper (Howard et al., 2019) reports V3-Large at 75.2% top-1 and about 51 ms on a single Pixel-1 large core, versus MobileNetV2 at 72.0% and about 75 ms — a 3.2-point accuracy gain at lower latency, the headline NAS result. On the Pixel-3 the absolute numbers drop (faster silicon) but the *relative* ordering holds: V3 keeps its lead over V2 at matched latency. The h-swish optimization alone, with its hardware-friendly implementation, contributes about a 0.9-point accuracy gain at negligible latency cost.

**The redesigned head saving.** The same paper quantifies the head/tail surgery: moving the final expansion past the average pool and trimming the initial conv from 32 to 16 filters together save roughly 7 to 10 ms on the Pixel — on the order of 15% of V3-Large's latency — with no accuracy loss. This is a reminder that a meaningful fraction of edge wins come not from clever blocks but from removing waste a profiler reveals, the kind of work the [edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle) post frames as routine practice.

**MobileNet as a quantization baseline.** Across the quantization literature MobileNetV2 int8 is the standard stress test, precisely because depthwise-separable networks are *harder* to quantize than fat networks — the depthwise layers have few weights per channel, so per-tensor quantization can blow up the dynamic range. Per-channel weight quantization recovers most of the loss: TensorFlow Lite reports MobileNetV2 dropping under 1 point top-1 going to int8 with per-channel weights, and Jacob et al. (2018), the integer-quantization paper, used MobileNet as the demonstration that 8-bit integer inference is production-viable. The lesson generalizes: if your network quantizes badly, suspect the depthwise layers first and reach for per-channel.

**On-device CV at scale.** MobileNet variants underpin a great deal of shipped mobile vision — on-device image classification, the backbone for detectors like SSD-MobileNet, segmentation heads, and feature extractors in AR pipelines. The reason is not that they are the most accurate networks; it is that they hit a latency-accuracy point that keeps a camera preview smooth at 30 FPS on commodity phones while leaving thermal and battery headroom for the rest of the app. That is the practical definition of "the right backbone."

**MobileNet as a transfer-learning backbone.** In practice most teams do not train MobileNet from scratch on ImageNet; they take the pretrained backbone, lop off the 1000-class classifier, and fine-tune a small new head on their own data — a few thousand images of their actual classes. The depthwise-separable features transfer well, fine-tuning is fast (often minutes on a single GPU because the body is small), and the resulting model inherits all of MobileNet's edge-friendliness. This is why a four-class doorbell or a defect-detector on a factory line so often starts from a MobileNet checkpoint: you get a deployable, quantizable, NPU-ready model for the price of a short fine-tune. When you do this, freeze the early blocks and fine-tune only the late blocks plus the new head if your dataset is small, to avoid overfitting the cheap features that already generalize.

**Composing MobileNet with the other levers.** MobileNet is the *architecture* lever, but it stacks with the other three. Quantization on top of MobileNet is the canonical int8 win covered above. Distillation can lift a small MobileNet's accuracy by training it against a larger teacher — a V3-Small student distilled from a ResNet teacher routinely recovers a point or two over training the student alone. And structured pruning (NetAdapt is itself a form of this) trims the channel counts further once you know your exact latency target. The right edge model is rarely "just MobileNet" — it is MobileNet, quantized, possibly distilled, with channel counts tuned to the device, all read off the same Pareto frontier the [taxonomy post](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) lays out.

## When MobileNet is the right backbone, and when it is not

MobileNet is the default, not the answer to everything. Reach for it when your target is a mobile or embedded device doing real-time vision on a tight latency and power budget, when you are CPU- or NPU-bound rather than accuracy-bound, and when you want a well-supported, quantization-friendly backbone with int8 kernels available in every runtime. V3-Small for the tightest budgets (wearables, low-end phones, even some microcontrollers with enough SRAM for the inverted-residual peak), V3-Large when you have a bit more headroom and want to maximize accuracy per millisecond, V2 when you need the broadest runtime compatibility (it is the most universally supported and the cleanest to quantize).

![Decision tree for choosing a MobileNet variant by latency budget, branching to V3-Small, V3-Large, or a successor architecture](/imgs/blogs/the-mobilenet-family-8.png)

The decision tree above encodes the workflow. Start from your *latency budget on the device* — not your accuracy target, because on the edge latency is the hard constraint and accuracy is what you maximize within it. Under about 20 ms (a microcontroller or a low-end phone needing real-time), reach for V3-Small or a thinned V2 at $\alpha = 0.5$. In the 20-to-60 ms mid-range, V3-Large quantized to int8 gives you about 75% top-1 in a few milliseconds. And if accuracy is genuinely the blocker and you have latency headroom, that is the signal to step up to a successor.

When is MobileNet *not* the right choice? When accuracy is the binding constraint and you have compute to spare — at that point MobileNet's ceiling (mid-70s top-1) is the limit, and you want a network that spends more compute better. The natural successor is **EfficientNet** (Tan and Le, 2019), which took MobileNetV2's inverted-residual block as its base unit and added *compound scaling*: instead of tuning width, depth, and resolution independently, scale all three together by a principled ratio. EfficientNet pushed past 84% top-1, and the [EfficientNet, ShuffleNet, and the FLOPs-latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap) post in this series digs into why its low FLOPs do not always translate to low latency — the same FLOPs-versus-latency tension that drove V3 to NAS. And when the task wants the global receptive field of attention — fine-grained recognition, some detection regimes — the successor is a hybrid like **MobileViT**, which interleaves MobileNet-style convolutions with lightweight transformer blocks; the [efficient attention and vision transformers for edge](/blog/machine-learning/edge-ai/efficient-attention-and-vision-transformers-for-edge) post covers that lineage. MobileNet is the foundation those successors build on, which is exactly why understanding it deeply pays off even when you eventually move past it.

A final caution that ties back to the series' core lesson. MobileNet's low FLOPs are real, but FLOPs are a proxy, and a leaky one. Depthwise convolutions are memory-bound — their arithmetic intensity is low, so on a device whose bottleneck is memory bandwidth rather than compute, a "cheap" MobileNet layer can be slower than its FLOP count suggests, and a "heavier" network with better-shaped operations can win. This is not a flaw in MobileNet; it is the reason V3 abandoned FLOP optimization for measured-latency NAS. When you deploy, *profile on the device* and trust the wall clock over the FLOP count. The [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) post gives you the framework to reason about which regime you are in, and the [capstone playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) stitches architecture choice, quantization, and profiling into one end-to-end workflow.

## Key takeaways

- **Depthwise-separable convolution is the foundation.** Splitting a standard convolution into a per-channel depthwise filter plus a $1\times1$ pointwise mixer cuts compute by about $1/K^2$ — roughly 8 to 9x for $3\times3$ kernels — and everything in the MobileNet family builds on that one factorization.
- **The width multiplier $\alpha$ and resolution multiplier $\rho$ each scale compute quadratically.** $\text{FLOPs} \approx \alpha^2 \rho^2 \cdot \text{FLOPs}_{\text{base}}$. $\alpha$ also scales parameters; $\rho$ does not. Together they turn one design into a whole accuracy-versus-cost family on a single curve.
- **V2's inverted residual gives depthwise room to work, and the linear bottleneck keeps the residual clean.** Expand thin to wide, filter cheaply in the wide space, project back down with no activation — because a ReLU on a thin tensor destroys information irreversibly.
- **V2's narrow-wide-narrow shape lowers peak inference memory.** Only the thin bottlenecks need to persist; the fat intermediate can be computed in slices. On SRAM-bound devices this is often the difference between fitting and not.
- **V3's win is optimizing the right objective.** Platform-aware NAS rewards *measured on-device latency*, not FLOPs, so it beats hand-designed V2 by about 3 points at lower latency — a gap you cannot close by sliding along V2's width-multiplier curve.
- **h-swish replaces swish's exponential with a clamped ReLU6 ramp.** $\text{h-swish}(x) = x \cdot \text{ReLU6}(x+3)/6$ — near-identical accuracy, no `exp()`, and a bounded range that quantizes cleanly. Use it in deep layers, ReLU in shallow ones.
- **MobileNet is the canonical quantization-friendly network, but watch the depthwise layers.** Bounded activations make PTQ to int8 easy (about 3.9x smaller, 2 to 4x faster, under a point of accuracy loss), but per-tensor quantization can blow up depthwise dynamic range — reach for per-channel weights.
- **FLOPs are a proxy; profile on the device.** Depthwise convolutions are memory-bound, so a low-FLOP model can be latency-bound by bandwidth. Start your variant choice from the *latency budget*, then maximize accuracy within it.
- **Know the successors.** When MobileNet's accuracy ceiling is the blocker, EfficientNet's compound scaling and MobileViT's hybrid attention are the next steps — both built on MobileNet's inverted-residual block.

## Further reading

- Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, Adam. **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** (2017) — the V1 paper, depthwise-separable backbone and the $\alpha$ / $\rho$ multipliers.
- Sandler, Howard, Zhu, Zhmoginov, Chen. **MobileNetV2: Inverted Residuals and Linear Bottlenecks** (2018) — the inverted residual, linear bottleneck, and memory-efficient inference argument.
- Howard, Sandler, Chu, Chen, Chen, Tan, Wang, Zhu, Pang, Vasudevan, Le, Adam. **Searching for MobileNetV3** (2019) — platform-aware NAS plus NetAdapt, h-swish, squeeze-and-excite, and the head/tail redesign.
- Tan, Chen, Pang, Vasudevan, Sandler, Howard, Le. **MnasNet: Platform-Aware Neural Architecture Search for Mobile** (2019) — the latency-in-the-reward NAS that V3 builds on.
- Jacob et al. **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference** (2018) — the integer-quantization paper that used MobileNet as its demonstration.
- TensorFlow Lite / LiteRT post-training quantization docs and the PyTorch `torch.ao.quantization` documentation — the practical int8 toolchains referenced above.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [building blocks for efficient models](/blog/machine-learning/edge-ai/building-blocks-for-efficient-models), [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq), [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas), [EfficientNet, ShuffleNet, and the FLOPs-latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap), and the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
