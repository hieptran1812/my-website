---
title: "Building blocks for efficient models: depthwise-separable, inverted residuals, and SE"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive the FLOP and parameter math for the cheap convolutional primitives that power every mobile vision model, implement them in PyTorch, and learn why an 8x FLOP cut rarely buys an 8x speedup."
tags:
  [
    "edge-ai",
    "model-optimization",
    "efficient-architecture",
    "depthwise-separable",
    "mobilenet",
    "inference",
    "efficient-ml",
    "convolution",
    "squeeze-and-excite",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/building-blocks-for-efficient-models-1.png"
---

A few years ago I inherited a perfectly good image classifier that someone had trained in the cloud and then thrown over the wall to me with a one-line ticket: "make this run on the phone." It was a ResNet-50. On a desktop GPU it was a delight — 76% top-1 on ImageNet, runs in a couple of milliseconds, who cares. On the mid-range Android phone the product team actually shipped to, a single inference took just under 400 ms and pinned a CPU core hot enough that the OS started thermal-throttling after a dozen frames. The camera preview we were supposed to annotate in real time turned into a slideshow.

My first instinct was the instinct everyone reaches for: compress what I had. Quantize it to int8, prune the dead channels, maybe distill it into something smaller. Those are the levers the rest of this series is about, and they are real — quantization alone got me from 400 ms to about 150 ms. But 150 ms is still 6-7 frames per second, and I had spent two weeks getting there. The uncomfortable truth, which took me embarrassingly long to internalize, is that I was sanding down a model that was the wrong shape to begin with. ResNet-50 spends the overwhelming majority of its multiply-adds on dense 3x3 convolutions that mix every input channel into every output channel at every spatial position. That operation is gorgeous on a GPU with thousands of multiply units and enormous memory bandwidth. On a phone it is a tax you pay on every single pixel.

The teams who actually ship vision models to phones do not start with a ResNet and squeeze. They start with a fundamentally cheaper *operation* and build the whole network out of it. The same task — same input resolution, same number of output channels, comparable accuracy — done with a block that costs eight or nine times fewer multiply-adds. That is not a compression technique applied after the fact; it is a design decision made at the very first line of the model definition. And it is, dollar for dollar of engineering effort, the single biggest efficiency win available. Quantization gives you 4x on size and maybe 2-3x on speed. A good efficient backbone gives you an order of magnitude before you have quantized anything, and then you quantize *that*.

![A two-column comparison of a standard 3x3 convolution against a depthwise-separable block that splits the work into a depthwise spatial filter and a pointwise 1x1 channel mixer.](/imgs/blogs/building-blocks-for-efficient-models-1.png)

This post is about those building blocks. By the end you will be able to derive, from first principles and on a napkin, the FLOP and parameter cost of a standard convolution and of every cheap alternative that replaced it — depthwise-separable convolutions, grouped convolutions, the inverted residual with its linear bottleneck, squeeze-and-excite channel attention, and ShuffleNet's channel shuffle. You will be able to implement each one in a few lines of PyTorch and count its FLOPs with a profiler. And — this is the part most tutorials skip — you will understand why these blocks, despite slashing FLOPs by an order of magnitude, often deliver a far smaller wall-clock speedup, because they trade a compute-bound operation for a memory-bound one. That last point is where good edge engineers separate from people who just read the FLOP column off a table. This is the architecture lever in the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression): not shrinking an existing model but designing a cheap one from the start.

## Start with the cost of a standard convolution

You cannot reason about a cheaper operation until you can price the expensive one exactly. So let us nail down what a standard convolution actually costs, in both multiply-add operations and in parameters, because every efficient block is defined by how it beats these two numbers.

Take a convolutional layer with a square kernel of size $k \times k$ (think $k = 3$). It reads an input feature map with $C_{in}$ channels and writes an output feature map with $C_{out}$ channels. To keep the algebra clean, assume "same" padding so the output spatial dimensions match the input, $H \times W$. For every one of the $H \times W$ output spatial positions, and for every one of the $C_{out}$ output channels, the layer computes a weighted sum over a $k \times k$ window across all $C_{in}$ input channels. That is $k^2 \cdot C_{in}$ multiply-adds per output value.

Count them all up. The number of multiply-add operations (I will write MACs; one MAC is one multiply plus one accumulate, and people loosely call $2 \times$ MACs the "FLOP count") for a standard convolution is

$$
\text{MACs}_{\text{std}} = H \cdot W \cdot C_{in} \cdot C_{out} \cdot k^2 .
$$

A word on the MAC-versus-FLOP bookkeeping, because sloppiness here causes endless confusion when you compare papers. A multiply-accumulate is one fused multiply and one add; most papers (and the `ptflops`/`thop` tools) report MACs, but some report "FLOPs" meaning $2 \times$ MACs (counting the multiply and the add separately), and a few report "FLOPs" but actually mean MACs. A 569M-MAC MobileNet is sometimes quoted as a "1.1 GFLOP" model and sometimes as a "569M FLOP" model — same network, factor-of-two difference in the headline, purely a counting convention. Throughout this post I count MACs and say so. When you read a comparison table, the first thing to verify is which convention each row uses, or you will conclude one model is twice as expensive as another when they are identical. I will keep the ratios — the 8.8x, the 13x — because ratios are immune to the convention; only the absolute numbers shift.

The parameter count is the kernel tensor itself, which has shape $(C_{out}, C_{in}, k, k)$, plus a bias per output channel that we will ignore as a rounding error:

$$
\text{Params}_{\text{std}} = C_{in} \cdot C_{out} \cdot k^2 .
$$

Notice that the spatial size $H \times W$ multiplies the MAC count but not the parameter count — weights are shared across positions. That is the whole reason convolutions exist, and it is also why a layer can have modest parameters but enormous compute: a 3x3 conv with 256 in and 256 out channels has only about 590K parameters, but at a $56 \times 56$ feature map it does $56 \cdot 56 \cdot 256 \cdot 256 \cdot 9 \approx 1.85$ billion MACs. Parameters fit in a phone's flash; the MACs are what eat your latency budget and your battery.

Let me make those two numbers concrete with a single running example I will keep using for the rest of the post. Picture a mid-network layer in a typical backbone: a $14 \times 14$ feature map (this is the resolution after a few downsampling stages on a $224 \times 224$ input) with $C_{in} = C_{out} = 512$ channels and a $3 \times 3$ kernel. The standard conv costs

$$
\text{MACs}_{\text{std}} = 14 \cdot 14 \cdot 512 \cdot 512 \cdot 9 \approx 462\,\text{M MACs}
$$

with $512 \cdot 512 \cdot 9 \approx 2.36$M parameters. Roughly half a billion multiply-adds for one layer, and a real network stacks dozens of them. Hold onto the number 462M MACs; we are about to demolish it.

## Depthwise-separable: factor the convolution in two

Here is the key observation that started the whole efficient-architecture movement. A standard convolution does two genuinely distinct jobs at once, fused into a single weight tensor. First, it *filters spatially* — it looks at the $k \times k$ neighborhood and detects edges, textures, gradients. Second, it *mixes channels* — it combines information across all $C_{in}$ input channels to form each output channel. There is no law of nature that says these two jobs must be done together. What if we separate them?

A **depthwise-separable convolution** factors the operation into two cheaper layers:

1. A **depthwise convolution**: a $k \times k$ spatial filter applied to each input channel *independently*. Channel $c$ gets its own $k \times k$ kernel; there is no mixing across channels at all. The output has the same number of channels as the input, $C_{in}$.
2. A **pointwise convolution**: an ordinary $1 \times 1$ convolution that mixes the $C_{in}$ channels into $C_{out}$ output channels at every spatial position, with no spatial extent.

Together they reproduce the *function* of a standard conv — spatial filtering followed by channel mixing — but at a fraction of the cost. Let us price each piece.

The depthwise step applies one $k \times k$ kernel per channel, across $H \times W$ positions:

$$
\text{MACs}_{\text{dw}} = H \cdot W \cdot C_{in} \cdot k^2 .
$$

The pointwise step is a $1 \times 1$ conv with $C_{in}$ in and $C_{out}$ out, so its kernel area is $1$:

$$
\text{MACs}_{\text{pw}} = H \cdot W \cdot C_{in} \cdot C_{out} .
$$

The total for the separable block is the sum:

$$
\text{MACs}_{\text{sep}} = H \cdot W \cdot C_{in} \cdot k^2 + H \cdot W \cdot C_{in} \cdot C_{out} = H \cdot W \cdot C_{in} \,(k^2 + C_{out}) .
$$

Now divide to get the reduction factor — how much cheaper the separable block is than the standard one for the *same* input and output shapes:

$$
\frac{\text{MACs}_{\text{sep}}}{\text{MACs}_{\text{std}}}
= \frac{H W C_{in}(k^2 + C_{out})}{H W C_{in} C_{out} k^2}
= \frac{k^2 + C_{out}}{C_{out} \, k^2}
= \frac{1}{C_{out}} + \frac{1}{k^2} .
$$

That clean little expression, $\frac{1}{C_{out}} + \frac{1}{k^2}$, is the most important formula in this post, so let us read it carefully. For a $3 \times 3$ kernel, $\frac{1}{k^2} = \frac{1}{9} \approx 0.111$. The other term, $\frac{1}{C_{out}}$, is tiny whenever the layer is wide — at $C_{out} = 512$ it is $\frac{1}{512} \approx 0.002$, basically noise. So the reduction is dominated entirely by the kernel-area term:

$$
\frac{\text{MACs}_{\text{sep}}}{\text{MACs}_{\text{std}}} \approx \frac{1}{k^2} = \frac{1}{9}
\quad\Longrightarrow\quad \text{about } 8\text{–}9\times \text{ cheaper for } 3\times3.
$$

The original MobileNet paper (Howard et al., 2017) quotes this as roughly an 8x to 9x reduction, and now you can see exactly where it comes from. It is not magic and it is not a hyperparameter — it is the kernel area, $k^2 = 9$, minus a small correction. The figure below traces this decomposition visually, splitting the dense kernel into the depthwise and pointwise pieces.

The same reduction holds for parameters, which matters for flash budget and download size:

$$
\text{Params}_{\text{sep}} = C_{in} \cdot k^2 + C_{in} \cdot C_{out},
\qquad
\frac{\text{Params}_{\text{sep}}}{\text{Params}_{\text{std}}} = \frac{1}{C_{out}} + \frac{1}{k^2} .
$$

Identical factor. The block is roughly 8-9x smaller *and* 8-9x cheaper to run.

#### Worked example: our 462M-MAC layer, separated

Take the running example — $14 \times 14$, $C_{in} = C_{out} = 512$, $k = 3$. The standard conv was 462M MACs and 2.36M params. The separable version:

- Depthwise: $14 \cdot 14 \cdot 512 \cdot 9 \approx 0.90$M MACs.
- Pointwise: $14 \cdot 14 \cdot 512 \cdot 512 \approx 51.4$M MACs.
- Total: $\approx 52.3$M MACs.

The reduction is $462 / 52.3 \approx 8.8\times$, exactly what $\frac{1}{512} + \frac{1}{9} = 0.0020 + 0.111 = 0.113 \approx \frac{1}{8.8}$ predicted. Parameters drop from 2.36M to $512 \cdot 9 + 512 \cdot 512 \approx 0.27$M, an 8.7x cut. Notice something important about *where* the cost went: of the 52.3M MACs, the depthwise step is under 1M and the pointwise $1 \times 1$ is over 51M. In a depthwise-separable network the $1 \times 1$ convolutions, not the depthwise ones, are where almost all the compute lives. That fact will drive every optimization decision later — including why grouped pointwise convs and channel shuffle were invented.

### What the factorization gives up, and why it usually does not matter

It would be dishonest to present depthwise-separable as a free lunch. You are not computing the *same function* as a standard conv at lower cost; you are computing a *restricted* family of functions. A standard $3 \times 3$ conv can learn any linear map from the $9 \cdot C_{in}$ input values in a receptive window to each of the $C_{out}$ outputs — a full, dense $(C_{out}) \times (9 C_{in})$ weight matrix per output position. The separable factorization forces that big matrix to be a *product* of two structured matrices: a block-diagonal spatial part (the depthwise) and a spatially-uniform channel-mixing part (the pointwise). It is a low-rank-style constraint on the space of functions the layer can represent. Any standard conv whose weight tensor does not factor that way is simply unreachable by a separable block.

So why does accuracy barely move? Two reasons, both empirical and both worth knowing. First, the functions real networks learn are heavily redundant — the dense weight tensors in a trained standard conv turn out to be approximately low-rank along exactly the axes the separable structure exploits, which is the same redundancy that makes pruning and low-rank factorization work. The separable block bakes that redundancy into the architecture instead of discovering it during training. Second, you get to *stack more blocks* for the same FLOP budget. Eight separable blocks cost about what one dense block costs, and a deeper stack of restricted operations is more expressive than a single unrestricted one. Depth buys back most of what the factorization gives up. The roughly one-point ImageNet drop from a comparable dense network to MobileNetV1 is the net of these forces, and it is a price almost everyone pays gladly for the 8-9x.

There is a corollary that bites people: because each output channel of a separable block is a linear combination (the pointwise) of single-channel spatial filters (the depthwise), the block cannot learn a feature that requires *jointly* filtering several input channels with different spatial kernels before mixing — for instance, an oriented-edge detector that needs channel A filtered horizontally and channel B filtered vertically *before* they are combined. A standard conv can; a single separable block cannot, because the depthwise applies one kernel per channel and only then does the pointwise mix. In practice the network routes around this by using two blocks, but if you ever see a separable architecture underperform badly on a task with strong cross-channel spatial structure, this is the reason, and the fix is more depth or a few strategically placed dense convs.

### Receptive field and downsampling: where stride lives

One more accounting detail that matters for building real networks. In a standard conv, the stride (for downsampling) lives in the one conv. In a separable block you must decide *which* of the two sub-convs carries the stride, and the answer is always the depthwise. The depthwise conv has the spatial extent, so it is the natural place to subsample; the pointwise is $1 \times 1$ and striding it would just throw away pixels with no filtering. So a stride-2 separable block is: depthwise $3 \times 3$ stride 2 (halves $H$ and $W$), then pointwise $1 \times 1$ stride 1 (mixes channels at the reduced resolution). This also means the expensive pointwise runs at the *smaller* output resolution, which is a small bonus saving — downsample first, then mix channels, and the channel-mixing happens over fewer spatial positions. Networks like MobileNet deliberately order operations to push the costly $1 \times 1$s to lower resolutions for exactly this reason.

## Grouped convolution: the dial between dense and depthwise

Depthwise convolution is actually the extreme case of an older, more general idea: the **grouped convolution**, introduced way back in AlexNet (2012) as an engineering hack to split a model across two GPUs, then rediscovered as an efficiency lever.

A grouped convolution partitions the channels into $g$ equal groups and runs $g$ independent convolutions, each seeing only $C_{in}/g$ input channels and producing $C_{out}/g$ output channels. There is no mixing across groups within the layer. Cost-wise, you are doing $g$ convolutions, each $\frac{1}{g^2}$ the size of the full one (both input and output channels shrink by $g$), so the total is $\frac{g}{g^2} = \frac{1}{g}$ of the dense cost:

$$
\text{MACs}_{\text{grouped}} = \frac{H \cdot W \cdot C_{in} \cdot C_{out} \cdot k^2}{g} .
$$

The two endpoints of the $g$ dial are familiar:

- $g = 1$ is an ordinary dense convolution (one group sees everything).
- $g = C_{in}$ (with $C_{out} = C_{in}$) is a depthwise convolution — each channel is its own group, $\frac{1}{C_{in}}$ the cost, no cross-channel mixing.

So depthwise conv is "grouped conv taken to its limit." Intermediate values, like $g = 2, 4, 8, 32$, give you a tunable trade-off: more groups means cheaper but less cross-channel information flow. ResNeXt (2017) used grouped convs with a moderate $g$ ("cardinality") to get more representational power per FLOP than plain ResNet. The price of grouping is the same price depthwise pays in the extreme: channels stop talking to each other. A network built only of grouped convs would have $g$ parallel, never-communicating sub-networks, which is useless. You need *something* to mix the groups back together — and that something is either a $1 \times 1$ conv (the pointwise step) or, more cleverly, a channel shuffle, which we will get to.

## The inverted residual and the linear bottleneck

MobileNetV1 stacked depthwise-separable blocks and called it a day. It worked, but it left accuracy on the table, and MobileNetV2 (Sandler et al., 2018) found out why with one of the most elegant pieces of reasoning in efficient-model design. The fix is the **inverted residual block with a linear bottleneck**, and it is worth slowing down for because the intuition is genuinely deep.

Start with the ordinary residual block you know from ResNet. It is *wide-narrow-wide*: take a thick feature map, squeeze it down with a $1 \times 1$ to a thin bottleneck, do the expensive $3 \times 3$ on the thin tensor (cheap, because it is thin), then expand back out with another $1 \times 1$, and add the skip connection. The skip connects the two *wide* ends.

The inverted residual flips this to *narrow-wide-narrow*. The block lives on a thin bottleneck representation. Inside, it:

1. **Expands** the thin input with a $1 \times 1$ conv by an expansion factor $t$ (typically $t = 6$), blowing $C$ channels up to $tC$.
2. **Filters** that wide tensor with a cheap depthwise $3 \times 3$ (cheap *because* it is depthwise, even though it is wide).
3. **Projects** back down to a thin output with another $1 \times 1$.

And — this is the inversion — the skip connection now connects the two *thin* ends, the narrow bottlenecks, not the wide middles. The block is illustrated below as a vertical stack from input bottleneck through expansion, depthwise filtering, and the linear projection back down.

![A vertical stack diagram showing the inverted residual block expanding the bottleneck sixfold, filtering with a depthwise conv, and projecting back down with a linear 1x1 that skips ReLU.](/imgs/blogs/building-blocks-for-efficient-models-3.png)

Two questions immediately arise. Why invert it at all? And what is this "linear bottleneck" business?

**Why invert.** Memory traffic. The skip connection has to be stored in memory across the whole block (you need the input around to add it at the end). If the skip carries the *wide* tensor, as in a classic residual, that is a big tensor sitting in memory. If the skip carries the *thin* bottleneck, the long-lived tensor is small, and only the expanded wide tensor exists transiently inside the block, where the framework can keep it in cache or even fuse it away. On a memory-constrained device, keeping the persistent activations thin is a direct win on peak RAM. MobileNetV2's bottlenecks are genuinely narrow (16, 24, 32 channels in early stages), and that is deliberate.

**The linear bottleneck.** This is the subtle part. The final $1 \times 1$ projection that takes the wide tensor back down to the thin bottleneck has *no activation function* — it is linear. No ReLU. MobileNetV1 and ResNet would have put a ReLU there. MobileNetV2 deliberately removes it, and the paper's argument is a manifold argument worth internalizing.

ReLU is information-destroying: it zeroes out everything negative. When you have a wide, high-dimensional tensor, that is usually fine — the information is spread across enough dimensions that a ReLU clipping some of them still leaves the relevant structure recoverable in the surviving dimensions. But the bottleneck is *low-dimensional by design*. The hypothesis is that the "manifold of interest" — the actual useful information in the activations — lives in a low-dimensional subspace, and the thin bottleneck is just enough room to hold it. If you apply a ReLU on that thin tensor, you are clipping a low-dimensional representation, and once you collapse part of a low-dimensional manifold you cannot recover it; the information is gone. The paper shows toy experiments where applying ReLU in low dimensions badly corrupts a manifold that survives the same ReLU in high dimensions. So the rule is: **ReLU is safe in the wide expanded space and dangerous in the thin bottleneck.** Keep the nonlinearities (ReLU6, specifically) inside the expansion where the tensor is fat, and make the projection back to the bottleneck linear so it does not destroy the compressed representation.

That single change — dropping the last ReLU — bought MobileNetV2 about a point of ImageNet accuracy over an otherwise identical architecture with the ReLU left in. It costs nothing at inference. It is free accuracy from understanding the geometry of what your activations actually are.

#### Worked example: the expansion-ratio trade

Let us price the inverted residual and see how the expansion factor $t$ trades cost against capacity. Take a bottleneck with $C = 64$ channels at a $14 \times 14$ resolution, expansion $t = 6$, depthwise $k = 3$. The three sub-layers:

- **Expand** $1 \times 1$: $C \to tC = 64 \to 384$. MACs $= 14 \cdot 14 \cdot 64 \cdot 384 \approx 4.82$M.
- **Depthwise** $3 \times 3$ on $tC = 384$ channels: $14 \cdot 14 \cdot 384 \cdot 9 \approx 0.68$M.
- **Project** $1 \times 1$: $tC \to C = 384 \to 64$. MACs $= 14 \cdot 14 \cdot 384 \cdot 64 \approx 4.82$M.
- **Total:** $\approx 10.3$M MACs.

Now sweep the expansion factor. The two $1 \times 1$ convs each scale linearly with $t$ (they are $C \to tC$ and $tC \to C$), and the depthwise scales linearly with $t$ too (it runs on $tC$ channels). So the *entire block cost is essentially linear in $t$*:

| Expansion $t$ | Expanded ch | Block MACs | Relative cost | Typical use |
|---|---|---|---|---|
| 1 | 64 | ~1.7M | 0.17x | degenerate, weak |
| 3 | 192 | ~5.2M | 0.50x | tight FLOP budget |
| 6 | 384 | ~10.3M | 1.0x | MobileNetV2 default |
| 10 | 640 | ~17.1M | 1.66x | accuracy-first, diminishing |

The takeaway: $t = 6$ is the empirical sweet spot the MobileNetV2 authors landed on. Below $t = 3$ the expanded space is too cramped for the depthwise filter to do useful work and accuracy falls off a cliff; above $t = 6$ you are paying linearly more MACs for shrinking accuracy gains. The expansion factor is the single most important knob in the block, and unlike most architecture choices it has a clean, near-linear cost model you can budget against. If you have a FLOP target, you can solve for the $t$ that fits.

### The manifold argument, made a little more rigorous

The hand-wave version of the linear-bottleneck argument — "ReLU destroys information in low dimensions" — is correct but worth sharpening, because once you see the mechanism precisely you will never put a ReLU on a bottleneck again. The claim has two parts: a ReLU can be information-preserving, and the condition under which it is depends on dimensionality.

Take a set of activations that lie on (or near) a $d$-dimensional manifold, embedded in a higher-dimensional space of $m$ channels. A ReLU is the map $x \mapsto \max(0, x)$ applied per coordinate. Consider what it does to the manifold. For any coordinate that is positive across the whole manifold, ReLU is the identity there — no information lost. The danger is the coordinates that change sign across the manifold: ReLU collapses the negative half-line of each such coordinate to zero, folding two distinct inputs that differ only in a negative coordinate value onto the same output. That folding is irreversible.

Now the dimensionality argument. The MobileNetV2 paper observes that if the manifold of interest occupies a low-dimensional subspace of a *high*-dimensional activation space, then there is "room to spare" — the manifold can be situated so that within the relevant region, ReLU acts as a near-invertible map (a coordinate-wise identity on the active orthant), and the information survives. The paper formalizes a sufficient condition: if the input manifold lies in a low-dimensional subspace and you first apply a random linear expansion to a much higher dimension *before* the ReLU, then with high probability the ReLU is invertible on the manifold's image. In symbols, the recoverable case is roughly "expand to dimension $\gg d$, then ReLU," and the lossy case is "ReLU directly on dimension $\approx d$." This is exactly the structure of the inverted residual: the ReLU6 nonlinearities live in the *expanded* $tC$-dimensional space (high dimension, room to spare, ReLU is fine), and the projection back to the $C$-dimensional bottleneck (low dimension, no room) is kept *linear* so no ReLU ever acts on the cramped representation. The architecture is a direct mechanical implementation of the theorem.

You can sanity-check the failure mode yourself in ten lines: take a 2-D spiral, embed it via a random linear map into $n$ dimensions, apply ReLU, then project back with the pseudo-inverse and measure reconstruction error. At $n = 2$ or $n = 3$ the spiral comes back mangled — whole arms collapsed. At $n = 30$ it comes back nearly perfect. That experiment is essentially Figure 1 of the MobileNetV2 paper, and running it once is the fastest way to make the linear-bottleneck rule stick.

### Activation memory: the quiet reason inversion wins

FLOPs and parameters get all the attention, but on a memory-constrained device the number that often decides whether your model *fits* is peak activation memory — the largest amount of intermediate tensor data alive at any instant during a forward pass. This is where the "inverted" in inverted residual pays a second dividend that has nothing to do with compute.

Walk through the lifetime of tensors in the block. The input bottleneck (thin, $C$ channels) must stay alive across the whole block because the residual add at the end needs it. The expanded tensor ($tC$ channels) is created, consumed by the depthwise, and can be freed; the depthwise output ($tC$) is consumed by the projection and freed; the projection output ($C$) is the result. So the *long-lived* tensor — the one that pins memory for the block's whole duration — is the thin bottleneck, and the fat $tC$ tensors are transient. A classic residual block does the opposite: the skip carries the *wide* tensor, which must stay resident across the block.

#### Worked example: peak RAM, classic versus inverted

Take a $56 \times 56$ feature map, classic residual with 128-channel wide ends and a 32-channel bottleneck, fp16 activations (2 bytes). The classic block's skip holds the wide tensor: $56 \cdot 56 \cdot 128 \cdot 2 \approx 1.6$ MB resident for the whole block, plus the transient bottleneck tensors. Now the inverted residual on a 24-channel bottleneck with $t=6$ (144-channel expansion) at the same resolution: the resident skip is the *thin* tensor, $56 \cdot 56 \cdot 24 \cdot 2 \approx 0.30$ MB, and the wide $144$-channel tensor ($56 \cdot 56 \cdot 144 \cdot 2 \approx 1.8$ MB) exists only transiently inside the block, where a good runtime can overlap its lifetime with the freed input or even fuse the three convs to avoid materializing it in DRAM at all. On a microcontroller with a 256 KB SRAM tensor arena, that difference — what stays resident versus what is transient and fuseable — is frequently the line between a model that runs entirely in SRAM and one that thrashes to slow external flash. The inverted residual was designed by people who had measured peak memory, not just FLOPs.

## Squeeze-and-excite: channel attention for almost free

The blocks so far make convolutions cheaper. Squeeze-and-excite (SE), from Hu et al. (2018), does the opposite in spirit — it *adds* a little capacity, but so cheaply that it is almost always worth it, and it composes beautifully with the efficient blocks above. It is the reason MobileNetV3 and the EfficientNet family beat their predecessors at the same FLOP budget.

The idea is channel attention. A convolution produces $C$ output channels, each a feature detector. But not every channel is equally useful for every input — a "wheel" channel matters for a photo of a car and is irrelevant for a photo of a cat. SE learns to *recalibrate* the channels per input: scale up the channels that matter for this particular image and scale down the ones that do not. The genius is doing this for almost no compute, by first collapsing all the spatial information away.

The SE block has three steps, shown as a side branch in the figure below — it taps the feature map, computes a per-channel gate, and multiplies that gate back in:

1. **Squeeze.** Global-average-pool the $H \times W \times C$ feature map down to a $1 \times 1 \times C$ vector — one number per channel, summarizing "how strongly is this channel firing across the whole image." This throws away all spatial detail and keeps only channel-wise statistics.
2. **Excite.** Push that $C$-vector through a tiny two-layer bottleneck MLP: a fully-connected layer down to $C/r$ (the reduction ratio $r$ is typically 16), a ReLU, a fully-connected layer back up to $C$, and a sigmoid. The output is $C$ numbers in $[0, 1]$ — a learned gate per channel.
3. **Scale.** Multiply each channel of the original feature map by its gate. Channels with gate near 1 pass through; channels with gate near 0 are suppressed.

![A branching dataflow diagram showing squeeze-and-excite pooling the feature map to one value per channel, learning gates through a tiny two-layer bottleneck, then multiplying the gates back into the feature map.](/imgs/blogs/building-blocks-for-efficient-models-4.png)

Now the cost. This is the beautiful part. The two FC layers operate on a $C$-vector, *not* on the full $H \times W \times C$ tensor — the squeeze step already collapsed the spatial dimensions. So the excite MLP costs

$$
\text{MACs}_{\text{SE}} = C \cdot \frac{C}{r} + \frac{C}{r} \cdot C = \frac{2 C^2}{r} .
$$

There is no $H \times W$ factor. Compare that to the convolution it sits next to, which *does* have the $H \times W$ factor and is therefore vastly larger. For a $14 \times 14$ map with $C = 512$ and $r = 16$, the SE block costs $\frac{2 \cdot 512^2}{16} \approx 33$K MACs, against the pointwise conv's 51M MACs in the same block. SE adds well under 0.1% to that block's compute and a similarly tiny slice of parameters, and in exchange it typically buys 0.5 to 1 point of ImageNet accuracy. That is one of the best accuracy-per-FLOP deals in the whole literature, which is why SE (and its descendants like ECA and CBAM) shows up in essentially every modern efficient backbone.

One honest caveat: SE is global-average-pool plus two tiny matmuls, and global pooling is a *reduction* across the whole feature map, which is awkward for some streaming hardware and introduces a synchronization point. On most NPUs it is fine; on some DSP-style accelerators the pooling op falls back to a slower path. Always profile, but the FLOP math says SE is nearly free and the accuracy data says it earns its keep.

## Channel shuffle: making grouped convs talk

Back to the grouped-conv problem. Recall that in a depthwise-separable block, the $1 \times 1$ pointwise conv is where almost all the MACs live (51M out of 52M in our example). ShuffleNet (Zhang et al., 2018) asked the obvious next question: what if we make the pointwise conv grouped too, to cut *its* cost by $\frac{1}{g}$? That would attack the actual bottleneck.

The problem we already flagged: if both the depthwise and the pointwise are grouped, then channels in group 1 never, ever mix with channels in group 2. Stack a few such blocks and you have $g$ completely isolated sub-networks that never share information — a representational disaster. You have made the network cheap and stupid.

ShuffleNet's fix is a **channel shuffle**: a zero-cost, parameter-free permutation of the channels inserted between grouped convolutions. After a grouped conv produces its $g$ groups of outputs, the shuffle re-deals the channels so that the next grouped conv's groups each contain channels that came from *all* of the previous groups. The mechanism is exactly like riffling a deck split into $g$ piles: take one card from each pile in turn. Concretely, you reshape the channel dimension from $(g, C/g)$ to $(C/g, g)$, transpose, and flatten back — a pure reindexing with no multiplies, often free on hardware because it is a memory layout change. The figure below shows two grouped convs feeding a shuffle that re-mixes the groups before the next stage.

![A branching diagram showing grouped convolutions producing isolated groups, a free channel shuffle re-dealing the channels across groups, and the next grouped conv seeing fully mixed channels.](/imgs/blogs/building-blocks-for-efficient-models-5.png)

With the shuffle in place, ShuffleNet can use grouped $1 \times 1$ convs and still let information cross group boundaries over the course of a couple of layers. This attacks the pointwise bottleneck directly: a grouped pointwise with $g = 3$ or $g = 8$ cuts the dominant cost by 3x to 8x on top of what depthwise-separable already saved. ShuffleNet hit ResNet-class accuracy at a tiny fraction of the FLOPs, and it did so by being honest about *where the FLOPs actually were* — in the $1 \times 1$s — and going after them rather than the depthwise convs that everyone visually associates with "the expensive part."

There is a lesson here that generalizes far beyond ShuffleNet: optimize where the cost is, not where your intuition says it should be. The depthwise conv *looks* like the spatial, expensive operation. It is almost free. The boring $1 \times 1$ that "just mixes channels" is where your latency budget is going. Profile before you optimize, every time.

## The arithmetic-intensity caveat: FLOP savings are not speedups

Now the part that separates engineers who ship from engineers who quote spec sheets. Everything above is about reducing FLOPs. You would be forgiven for assuming that an 8.8x FLOP reduction yields an 8.8x speedup. It does not. On real edge hardware, a depthwise-separable backbone with 8-9x fewer FLOPs than its dense equivalent often runs only 2-3x faster. Sometimes the depthwise layer alone, in isolation, runs *slower per FLOP* than the dense conv it replaced. Understanding why is the most valuable thing in this post.

The reason is **arithmetic intensity** — the ratio of compute done to bytes of memory moved, measured in FLOPs per byte. Every operation has to read its inputs from memory and write its outputs back. If an operation does a lot of math per byte it touches, the hardware's multiply units stay busy and you are *compute-bound* — your speed is set by peak FLOP/s, and cutting FLOPs helps directly. If an operation does little math per byte, the multiply units sit idle waiting on memory, and you are *memory-bound* — your speed is set by memory bandwidth, and cutting FLOPs does nothing because you were never FLOP-limited in the first place. This is the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and it is the single most important diagnostic in edge optimization.

Here is the cruel irony of depthwise convolutions. A dense $3 \times 3$ conv has *high* arithmetic intensity: each weight is reused across many spatial positions and each input value is reused across many output channels, so it does $k^2 \cdot C_{in} = 9 \cdot 512 \approx 4600$ MACs for roughly every input value it reads. That is a lot of math per byte; it is comfortably compute-bound, and on a GPU it saturates the tensor cores. A depthwise conv, by contrast, has *terrible* arithmetic intensity: each input channel is touched by exactly one $k \times k$ kernel and produces exactly one output channel, so it does only $k^2 = 9$ MACs per input value. It reads almost as many bytes as a dense conv but does a fraction of the math. Its FLOPs-per-byte is low, which plants it firmly under the memory roof. The before-and-after below contrasts the FLOP reduction you were promised against the latency reduction the hardware actually delivers.

![A two-column comparison showing that an 8x FLOP reduction is predicted to give 8x speedup but the memory-bound depthwise stage delivers only about 2.5x in practice.](/imgs/blogs/building-blocks-for-efficient-models-7.png)

Let us put numbers on it. Arithmetic intensity $I$ in MACs per element read:

- Dense $3 \times 3$, $C_{in} = C_{out} = 512$: roughly $k^2 \cdot C_{out} / (\text{reads per output})$. A clean way to see it: total MACs $= H W C_{in} C_{out} k^2$, total bytes (weights + input + output, fp32) is dominated by activations $\approx H W (C_{in} + C_{out}) \cdot 4$. The ratio is on the order of $\frac{C_{in} C_{out} k^2}{(C_{in}+C_{out})} \approx \frac{512 \cdot 512 \cdot 9}{1024} \approx 2300$ MACs/element — extremely compute-bound.
- Depthwise $3 \times 3$, 512 channels: total MACs $= H W C \cdot 9$, activation bytes $\approx H W \cdot 2C \cdot 4$. The ratio is $\frac{C \cdot 9}{2C} = 4.5$ MACs/element — over two orders of magnitude lower. Deeply memory-bound.

So when you replace a dense conv with depthwise-separable, you cut the FLOPs of the dense conv by 8.8x, but you replace the high-intensity dense op with a *low-intensity* depthwise op plus a moderate-intensity pointwise op. The depthwise op's runtime is gated by memory bandwidth, which you did not improve at all. The result, on real hardware, is the gap the figure shows: an 8.8x FLOP cut buying maybe a 2-3x latency cut.

This is not a reason to avoid efficient blocks — a 2-3x real speedup is enormous and they also shrink model size and memory by the full 8-9x, which matters independently. But it is a reason to never, ever quote a FLOP reduction as if it were a speedup. The honest claim is "8.8x fewer FLOPs and 8.7x smaller, measured at roughly 2.5x faster on the target," and you only know that last number by measuring on the actual device. This is also why hardware vendors specifically optimize their NPU kernels and memory layouts for depthwise convolution: the operation is so memory-bound that kernel-level fusion (fusing the depthwise into the surrounding pointwise convs to avoid round-tripping the intermediate activation to DRAM) can matter more than the FLOP count. We will return to that fusion idea when the series gets to compilers.

There is a deeper design principle hiding here that the ShuffleNetV2 paper made explicit, and it changed how careful people design these blocks. The total runtime of a network is not just compute; it is compute *plus memory-access cost* (MAC, confusingly the same acronym — here it means megabytes of memory traffic). When you are memory-bound, the thing to minimize is bytes moved, and the byte count of a $1 \times 1$ conv is dominated by its input and output activation tensors: $\text{bytes} \approx (H W C_{in} + H W C_{out} + C_{in} C_{out})$. For a fixed FLOP budget $B = H W C_{in} C_{out}$, the activation traffic $H W (C_{in} + C_{out})$ is *minimized* when $C_{in} = C_{out}$ — equal input and output channels. This is a real, derivable design rule: a $1 \times 1$ conv that keeps the channel count constant moves the fewest bytes per FLOP, so it runs fastest per FLOP on memory-bound hardware. ShuffleNetV2 turned this and three sibling observations (group count raises memory cost, network fragmentation hurts parallelism, element-wise ops are not free) into concrete design guidelines, and the headline lesson is the one this whole section has been building toward: *design against measured latency and memory traffic, not against the FLOP column.* The blocks that win on a spec sheet and the blocks that win on the device are not always the same blocks, and the difference is arithmetic intensity.

#### Worked example: a balanced versus lopsided pointwise

Make the channel-balance rule concrete. Two $1 \times 1$ convs at $28 \times 28$, each constrained to the same $\approx 100$M-MAC budget. Block A is balanced: $C_{in} = C_{out} = 256$, giving $28 \cdot 28 \cdot 256 \cdot 256 \approx 51$M MACs and activation traffic of $28 \cdot 28 \cdot (256 + 256) \approx 0.40$M elements. Block B is lopsided: $C_{in} = 512$, $C_{out} = 128$, same $28 \cdot 28 \cdot 512 \cdot 128 \approx 51$M MACs but activation traffic of $28 \cdot 28 \cdot (512 + 128) \approx 0.50$M elements — 25% more bytes for the identical FLOP count. On a memory-bound device that 25% more traffic translates roughly to 25% more latency for the same compute. The balanced block is strictly better on memory-bound hardware at equal FLOPs, and you would never see this in a FLOP-only comparison. This is why well-tuned efficient backbones tend to keep channel counts steady within a stage and step them up only at downsampling boundaries.

## Implementing the blocks in PyTorch

Enough theory. Here is each block in idiomatic PyTorch. The depthwise convolution is not a special layer — it is a plain `nn.Conv2d` with `groups=in_channels`, which is the cleanest demonstration that depthwise is just grouped-conv at the limit.

```python
import torch
import torch.nn as nn

class DepthwiseSeparable(nn.Module):
    """Depthwise 3x3 (per-channel spatial) + pointwise 1x1 (channel mix)."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise: groups == in_ch means each channel gets its own kernel.
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        # Pointwise: ordinary 1x1 conv mixes channels, no spatial extent.
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x
```

The only non-obvious line is `groups=in_ch` in the depthwise conv. PyTorch's `Conv2d` divides both input and output channels by `groups`; setting `groups` equal to the channel count gives exactly one $k \times k$ kernel per channel with no cross-channel mixing. That single argument is the whole depthwise trick.

Now the inverted residual with its linear bottleneck. Watch the two deliberate choices: the expansion factor `t`, and the *absence* of an activation after the final projection.

```python
class InvertedResidual(nn.Module):
    """MobileNetV2 block: expand (1x1) -> depthwise (3x3) -> project (1x1, linear)."""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden = in_ch * expand_ratio
        # Use the residual skip only when shapes match (stride 1, same channels).
        self.use_skip = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            # Expand: 1x1 up to the wide hidden dimension, with ReLU6.
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            # Depthwise on the WIDE tensor (cheap because depthwise), with ReLU6.
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Project: 1x1 back DOWN to the thin bottleneck. NOTE: NO activation.
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_skip else out
```

The linear bottleneck is encoded by what is *not* in the list: there is no `nn.ReLU6` after the final `nn.Conv2d(hidden, out_ch, 1)`. If you add one there "for symmetry," you reintroduce the information-destroying ReLU on the low-dimensional manifold and you lose roughly a point of accuracy for free. This is the single most common bug I have seen people introduce when reimplementing MobileNetV2 from memory — they add the missing activation because every other conv in the network has one.

The squeeze-and-excite module slots in as a side branch you can drop into any block:

```python
class SqueezeExcite(nn.Module):
    """Channel attention: global pool -> FC bottleneck -> sigmoid gate -> scale."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)            # squeeze H,W -> 1,1
        self.fc1 = nn.Conv2d(channels, squeezed, 1)    # 1x1 conv == FC on the vector
        self.fc2 = nn.Conv2d(squeezed, channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)                 # B,C,1,1 : one number per channel
        s = self.act(self.fc1(s))        # reduce to C/r
        s = self.gate(self.fc2(s))       # expand to C, squash to (0,1)
        return x * s                     # broadcast-multiply gates back in
```

Implementing the FC layers as $1 \times 1$ convolutions on the pooled $C \times 1 \times 1$ tensor is a common idiom — it avoids a `view`/`reshape` round-trip and keeps the tensor in NCHW layout so the broadcast-multiply at the end is clean. The whole module is a global pool and two tiny matmuls, exactly as the FLOP math promised.

Finally, the channel shuffle is pure tensor reshaping — no learnable parameters at all:

```python
def channel_shuffle(x, groups):
    """Re-deal channels so the next grouped conv mixes across groups. Free."""
    b, c, h, w = x.shape
    assert c % groups == 0
    x = x.view(b, groups, c // groups, h, w)   # split channel dim into (g, c/g)
    x = x.transpose(1, 2).contiguous()         # swap to (c/g, g)
    return x.view(b, c, h, w)                   # flatten back; channels re-interleaved
```

## Counting FLOPs to prove the reduction

Math on a napkin is good; a profiler confirming it is better. Use `ptflops` or `thop` to count MACs and parameters for any module. Here is the standard-versus-separable comparison made measurable for our running layer ($14 \times 14$, 512 channels in and out).

```python
from ptflops import get_model_complexity_info
import torch.nn as nn

def count(module, shape):
    macs, params = get_model_complexity_info(
        module, shape, as_strings=False,
        print_per_layer_stat=False, verbose=False,
    )
    return macs, params  # macs here are multiply-accumulates

shape = (512, 14, 14)  # C, H, W

standard = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
separable = DepthwiseSeparable(512, 512)

s_macs, s_params = count(standard, shape)
d_macs, d_params = count(separable, shape)

print(f"standard : {s_macs/1e6:6.1f} M MACs  {s_params/1e6:5.2f} M params")
print(f"separable: {d_macs/1e6:6.1f} M MACs  {d_params/1e6:5.2f} M params")
print(f"reduction: {s_macs/d_macs:.1f}x MACs  {s_params/d_params:.1f}x params")
```

Running this prints numbers that match the hand derivation almost exactly:

```console
standard :  462.4 M MACs   2.36 M params
separable:   52.4 M MACs   0.27 M params
reduction:   8.8x MACs   8.7x params
```

There is the 8.8x, confirmed by a tool rather than asserted. Note that `ptflops`'s depthwise-separable number includes the BatchNorms (negligible) and the activations (free), and it counts the depthwise and pointwise stages separately if you ask for the per-layer breakdown — which is how you would confirm, in your own model, that the $1 \times 1$ pointwise conv is eating 98% of the block's MACs.

To measure *latency* rather than FLOPs — which is the number that actually matters, given everything we said about arithmetic intensity — wrap each module in a proper timing harness with warm-up and many iterations. Never time a single forward pass; the first few are dominated by allocation and kernel compilation, and a single sample tells you nothing about the distribution.

```python
import torch, time

@torch.no_grad()
def bench(module, shape, device="cpu", warmup=20, iters=200):
    module = module.to(device).eval()
    x = torch.randn(1, *shape, device=device)   # batch=1 is the edge reality
    for _ in range(warmup):                      # warm up caches + kernels
        module(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        module(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms per inference

print(f"standard : {bench(standard,  shape):.3f} ms")
print(f"separable: {bench(separable, shape):.3f} ms")
```

On a typical laptop CPU at batch size 1, the separable block does *not* come out 8.8x faster. You will see something closer to 2-4x, with the exact ratio depending wildly on whether your BLAS backend has a well-tuned depthwise kernel. That gap between the 8.8x FLOP ratio and the measured latency ratio is the arithmetic-intensity caveat made real on your own machine. It is worth running yourself once — seeing the FLOP number and the latency number disagree on your own hardware is the moment the roofline stops being abstract.

## Results: how the blocks stack up

Pulling it together, here is the comparison that should live in your head when you choose a block. The numbers are for a representative mid-network layer at $14 \times 14$ with 512 channels (or its inverted-residual equivalent), normalized to the standard conv. Accuracy impact is the rough ImageNet top-1 delta reported across the MobileNet/ShuffleNet/SE literature when the block is swapped into a comparable architecture; treat these as approximate, directional figures, not guarantees for your task.

| Block | Rel. MACs | Rel. params | Arithmetic intensity | Accuracy impact | When to use |
|---|---:|---:|---|---:|---|
| Standard 3x3 conv | 1.00x | 1.00x | High (compute-bound) | baseline | Stems, GPU/server, when bandwidth is plentiful |
| Depthwise-separable | 0.11x | 0.11x | Low (depthwise memory-bound) | ~ -1 pt | Mobile backbones (MobileNetV1) |
| Grouped conv (g=4) | 0.27x | 0.27x | Medium | ~ -0.5 pt | More capacity/FLOP than depthwise (ResNeXt) |
| Inverted residual (t=6) | ~0.10x | ~0.13x | Low (depthwise stage) | ~ +0.5 pt | Default mobile block (MobileNetV2/V3) |
| + Squeeze-excite | +<0.001x | +small | unchanged conv path | ~ +0.5 to +1 pt | Cheap accuracy bolt-on (V3, EfficientNet) |
| ShuffleNet unit (g=3) | ~0.04x | ~0.05x | Low | comparable | Extreme FLOP budgets |

The matrix figure below renders the four core blocks against FLOPs, params, intensity, and accuracy so the trade-offs read at a glance.

![A comparison matrix of standard, depthwise-separable, grouped, and inverted-residual blocks across relative FLOPs, parameters, arithmetic intensity, and accuracy impact.](/imgs/blogs/building-blocks-for-efficient-models-6.png)

The one column people skip is "arithmetic intensity," and it is the column that explains why two blocks with the same FLOP count can have very different latency. A grouped conv with $g = 4$ has 2-3x the FLOPs of depthwise-separable but *higher* arithmetic intensity, so on a bandwidth-starved device it can actually run at a comparable or even better latency despite the higher FLOP count, because it keeps the compute units busier. FLOPs are a planning number. Latency is the truth, and arithmetic intensity is the bridge between them. The reduction-factor matrix below shows precisely where the depthwise-separable savings come from as the kernel size changes.

![A matrix showing how the depthwise-separable reduction factor grows with kernel size, dominated by the one-over-k-squared term.](/imgs/blogs/building-blocks-for-efficient-models-2.png)

#### Worked example: a full backbone, before and after

Let me ground the whole post in one shipped-scale result. Replace the dense $3 \times 3$ convolutions in a ResNet-style backbone with inverted residuals and you go from roughly 4 billion MACs (ResNet-50 territory) to roughly 300 million MACs (MobileNetV2 territory) at $224 \times 224$ — about a 13x FLOP cut — while ImageNet top-1 drops only from about 76% to about 72%. Model size falls from roughly 98 MB (fp32) to roughly 14 MB. On a mid-range mobile CPU, measured single-image latency falls from the high-hundreds of milliseconds to the tens of milliseconds — call it from around 400 ms to around 60-80 ms, a 5-6x real speedup, not the full 13x the FLOP cut would suggest, exactly because so much of the new compute is memory-bound depthwise work. That is the architecture lever doing in one design decision what weeks of quantization and pruning on the original ResNet could not: an order of magnitude on size, most of an order of magnitude on latency, for a few points of accuracy you can often claw back with training tricks or by going to MobileNetV3 with SE. And then you quantize the MobileNet on top of that for another 4x on size and 2x on speed. The levers compose; the architecture lever just happens to be the biggest and the cheapest to pull, because you pull it once at design time.

For the rules of thumb on which numbers to trust here — why latency and not FLOPs, why batch=1, why warm-up matters — see [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device).

## Training and quantizing these blocks: the gotchas

Designing the block is half the job; the blocks have personalities at training and quantization time that will surprise you if you have only ever trained dense ResNets. A few hard-won notes.

**Depthwise convs train differently.** Each depthwise channel is a tiny, isolated $k \times k$ filter with only nine weights and its own slice of the activation statistics. They are far more sensitive to learning rate and to weight decay than the fat $1 \times 1$s — apply heavy weight decay to the depthwise kernels and you will regularize them into uselessness, because there is so little to regularize. The common practice (and the MobileNet training recipes) is to use little or no weight decay on depthwise filters while keeping normal decay on the pointwise convs. They also benefit from their own BatchNorm, which is why every block above has a `BatchNorm2d` after both the depthwise and the pointwise. The normalization matters more here than in a dense net because the per-channel depthwise outputs have wildly different scales with nothing to average them out across channels.

**ReLU6, not ReLU.** Notice every activation in the blocks is `ReLU6` — $\min(\max(0, x), 6)$ — not plain ReLU. The cap at 6 is not arbitrary. It bounds the activation range, which makes the subsequent int8 quantization far better behaved: a known, fixed range means a tight, stable quantization scale, instead of chasing a long tail of large activations that waste quantization levels. When you build a model intended for int8 deployment, bounded activations like ReLU6 (or hard-swish in V3) are a quantization-friendliness decision baked into the architecture, made years before the quantizer runs.

**The linear bottleneck is a quantization minefield, and a gift.** The projection output has no ReLU, so it carries *signed* values with a potentially wide range, and it is immediately added to the residual. That add of two int8 tensors with different scales is exactly the kind of thing that goes wrong in post-training quantization and shows up as a mysterious accuracy cliff. The standard fix is per-tensor (or better, per-channel) quantization parameters tuned with a proper calibration set, and treating the residual add as its own quantized op with a rescale. The gift side: because the bottleneck is linear and low-dimensional, the *amount* of information you are quantizing there is small, so once the scales are right it quantizes cleanly. The cross-link to make here is the full int8 story — these blocks are the canonical thing you quantize, and the activation-range design above is what makes that quantization land.

#### Worked example: stress-testing the block at int4 and on a fallback NPU

Pose the real engineering problem. You have shipped a MobileNetV2-with-SE backbone at int8 on a phone NPU, hitting 71.5% top-1 at 12 ms p50. Two pressures arrive. First, marketing wants the model on a cheaper device with a smaller NPU, and the obvious lever is going to int4 weights. Second, the cheaper device's NPU does not implement the global-average-pool that SE needs, so that op falls back to the CPU.

Reason it through. Int4 on the *pointwise* $1 \times 1$ convs — which hold most of the parameters and compute — is usually survivable with per-channel quantization and a little quantization-aware fine-tuning; expect a drop of a point or two, recoverable. Int4 on the *depthwise* convs is dangerous: each depthwise channel has only nine weights, so there is almost no statistical room to absorb quantization error, and a four-bit grid on a nine-weight filter can wipe out the filter's selectivity. The decision: keep depthwise at int8 (it is a rounding error in size anyway) and push only the pointwise to int4. This is mixed-precision applied at the block level, and it follows directly from the block's structure.

Now the SE fallback. The global pool runs on CPU, which means every SE block forces an NPU-to-CPU-to-NPU round trip — a synchronization stall that, measured, can cost more wall-clock than the entire convolutional path of the block, even though SE is under 0.1% of the FLOPs. The stress test exposes that a "nearly free" op by FLOP count can be expensive by *latency* when it does not map to the accelerator. The fix is not to delete SE (it is buying you accuracy) but to either find an NPU that supports pooling, fuse the pool into a supported reduction op, or replace SE with a variant (like ECA) that uses a 1-D conv the NPU does support. The lesson generalizes: FLOP-cheap is not the same as hardware-cheap, and the only way to know is to profile the op on the actual fallback path.

For the full quantization pipeline that these architectural choices feed into — calibration, per-channel scales, the int8 conversion flow — and the mixed-precision decisions sketched above, the rest of this series goes deep; this post is about giving the quantizer a model that is already the right shape.

## How these blocks compose into families

None of these blocks is used alone. They are the alphabet; the MobileNet, ShuffleNet, and EfficientNet families are the words. A quick map of how they assemble, so you can read any modern efficient architecture:

**MobileNetV1** is the simplest: a stem of one ordinary conv, then a long stack of depthwise-separable blocks with periodic stride-2 downsampling, then a global pool and classifier. Two global hyperparameters — a width multiplier $\alpha$ that scales every channel count and a resolution multiplier $\rho$ that scales the input size — let you slide along the accuracy-latency curve without redesigning anything. Want it smaller? Set $\alpha = 0.5$ and every layer halves its channels, quartering the FLOPs.

**MobileNetV2** replaces the plain separable block with the inverted residual and linear bottleneck. Same overall shape — stem, stack of blocks, head — but each block is the expand-depthwise-project sandwich with a thin-to-thin skip. This is the architecture most "mobile backbone" defaults still point at.

**MobileNetV3** adds two things on top of V2: squeeze-and-excite in selected blocks, and a hardware-friendly activation (hard-swish, a cheap piecewise approximation of swish that avoids expensive exponentials on integer hardware). It also used neural architecture search to tune the per-stage expansion factors and which blocks get SE. The result is a measurable accuracy-latency improvement over V2 at the same budget — SE earning its near-free keep, exactly as the FLOP math predicted.

**ShuffleNet** goes after the pointwise bottleneck with grouped $1 \times 1$ convs plus channel shuffle, trading a little cross-channel bandwidth for the lowest FLOPs in the family. It shines at the extreme low end where every MAC counts.

**EfficientNet** is the synthesis: inverted residuals with SE as the base block, and a principled "compound scaling" rule that grows depth, width, and resolution together rather than one at a time. It is the clearest demonstration that the blocks in this post, combined with disciplined scaling, define a Pareto frontier you can dial along.

The decision tree below summarizes which block to reach for given your target and stage — and the full architectural stories live in the dedicated posts. For the MobileNet line in detail, see [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family); for the ShuffleNet and EfficientNet comparison and the all-important FLOPs-versus-latency gap on real silicon, see [EfficientNet, ShuffleNet, and the FLOPs-latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap).

![A decision tree mapping the target and network stage to the right block: dense convs for GPUs and stems, inverted residual with SE for mobile, grouped with shuffle for microcontrollers.](/imgs/blogs/building-blocks-for-efficient-models-8.png)

## When to reach for which block (and when not to)

Decisive recommendations, because every choice is a cost:

**Use a standard dense conv** in two places even in an "efficient" model: the stem (the first one or two layers, where channel counts are tiny so the dense cost is small and the high arithmetic intensity is welcome), and anywhere you are targeting a bandwidth-rich accelerator like a server GPU. On a GPU with tensor cores and hundreds of GB/s of bandwidth, dense convs run at near-peak efficiency and depthwise convs *waste* the hardware — they leave the tensor cores idle. Do not reflexively port a mobile architecture to a GPU; you may be slower. This is the most common mistake I see: an "efficient" model that is efficient only on the device it was designed for.

**Use depthwise-separable or inverted residuals** for the bulk of a mobile or NPU backbone. This is the default and it is the right default. Prefer the inverted residual (MobileNetV2/V3 style) over plain separable (V1 style) — the linear bottleneck and residual skip are free accuracy. Pick the expansion factor by your FLOP budget; $t = 6$ unless you are tight, then $t = 3$ or $t = 4$.

**Add squeeze-and-excite** almost always, unless your target hardware specifically chokes on global pooling (some streaming DSPs do). It is the cheapest accuracy you will ever buy. Reduction ratio $r = 16$ is a fine default; smaller $r$ (more capacity in the gate) for accuracy-first, larger $r$ for the tightest budgets.

**Reach for grouped convs (not depthwise) and channel shuffle** only at the extreme low end — microcontroller-class budgets, sub-50M-FLOP models — or when you have measured that your depthwise kernels are so memory-bound that a higher-intensity grouped conv is actually faster on your silicon. Grouped convs are a more nuanced tool; do not start there.

**Do not assume FLOPs equal latency, ever.** The whole back half of this post was one long argument for this rule. An efficient block can be slower per FLOP than the dense one it replaced because it traded compute-bound for memory-bound work. Always measure on the actual target, at batch size 1, with warm-up. If your profiler says the depthwise layers are memory-bound (and it will), the highest-leverage optimization is no longer fewer FLOPs — it is operator fusion to stop round-tripping activations to DRAM, which is a compiler and runtime concern.

**Do not apply ReLU on a thin bottleneck.** The linear-bottleneck argument is real. If you are designing or reimplementing a block, the projection back to a narrow representation must be linear. Putting a nonlinearity there is a silent accuracy bug.

## Case studies and real numbers

A few load-bearing results from the literature, so this is not just my own benchmarks:

**MobileNetV1 (Howard et al., 2017).** The paper that introduced depthwise-separable convolutions to mainstream vision. Their headline: roughly 8-9x fewer multiply-adds than a comparable VGG-style dense network, with MobileNet-224 at $\alpha=1.0$ hitting about 70.6% ImageNet top-1 at roughly 569M MACs and 4.2M parameters — versus a dense network needing several times the compute for similar accuracy. The width and resolution multipliers gave a clean family of models from about 41M to 569M MACs.

**MobileNetV2 (Sandler et al., 2018).** Introduced the inverted residual and linear bottleneck. MobileNetV2 at $\alpha=1.0$ reached about 72.0% top-1 at roughly 300M MACs and 3.4M parameters — beating V1 on both accuracy *and* efficiency. Their ablation specifically isolated the linear-bottleneck contribution (removing the last ReLU) at roughly a point of top-1, which is the cleanest published evidence for the manifold argument.

**SE-Net (Hu et al., 2018).** Won the final ImageNet classification challenge. Adding SE blocks to ResNet-50 improved top-1 by roughly 1 point (from about 76.3% to about 77.6%) while increasing compute by well under 1% and parameters by a few percent — the canonical demonstration that channel attention is nearly free. The same module dropped into MobileNetV3 contributed materially to its V2-over gain.

**ShuffleNet (Zhang et al., 2018).** Pushed the low-FLOP frontier with grouped pointwise convs plus channel shuffle. ShuffleNet at roughly 140M MACs reached accuracy competitive with much larger models, and — relevant to our caveat — the paper itself discusses that grouped convs and shuffles must be evaluated by actual inference time on the target, not just FLOPs, because memory-access cost dominates at that scale. ShuffleNetV2 went further and proposed direct latency-based design guidelines, explicitly rejecting FLOPs as the sole proxy.

The through-line across all four: each paper made the architecture cheaper *and* then had to confront that FLOPs alone did not tell the latency story. That tension is the subject of this whole series.

## Key takeaways

- The biggest efficiency win is architectural, and it is made at design time, not by compressing a finished model. A good efficient backbone beats a compressed dense one before you have quantized anything.
- A standard conv costs $H W C_{in} C_{out} k^2$ MACs and does two jobs at once: spatial filtering and channel mixing. Every efficient block beats it by factoring those jobs apart.
- Depthwise-separable splits the conv into a per-channel depthwise filter plus a $1 \times 1$ pointwise mixer, for a reduction of $\frac{1}{C_{out}} + \frac{1}{k^2}$ — about 8-9x for a $3 \times 3$ kernel, dominated by the kernel-area term.
- In a separable block almost all the MACs live in the $1 \times 1$ pointwise conv, not the depthwise one. Optimize where the cost is.
- Grouped conv ($\frac{1}{g}$ cost) is the dial between dense ($g=1$) and depthwise ($g=C$); channel shuffle lets stacked grouped convs share information for free.
- The inverted residual expands then projects, keeps the persistent skip tensor thin to save memory, and makes the final projection *linear* so a ReLU does not destroy the low-dimensional bottleneck manifold.
- Squeeze-and-excite is channel attention for under 0.1% extra MACs, because it pools away the spatial dimensions before the tiny FC gate. Add it almost always.
- An 8x FLOP cut is not an 8x speedup. Depthwise convs are memory-bound (low arithmetic intensity), so measured latency improves far less than FLOPs. Always measure on the real target at batch=1.
- Dense convs still win on bandwidth-rich GPUs and in the stem; depthwise/inverted residuals own mobile backbones; grouped+shuffle owns the microcontroller extreme.

## Further reading

- Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017) — the depthwise-separable paper and the reduction-factor derivation.
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018) — the inverted residual and the manifold argument for the linear bottleneck.
- Hu et al., "Squeeze-and-Excitation Networks" (2018) — cheap channel attention, the ImageNet 2017 winner.
- Zhang et al., "ShuffleNet: An Extremely Efficient CNN for Mobile Devices" (2018) and ShuffleNetV2 (2018) — grouped pointwise convs, channel shuffle, and direct-latency design guidelines.
- Tan and Le, "EfficientNet: Rethinking Model Scaling for CNNs" (2019) — compound scaling over inverted-residual-plus-SE blocks.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for the arithmetic-intensity caveat, [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for measuring latency honestly, [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) and [EfficientNet, ShuffleNet, and the FLOPs-latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap) for the full architectures, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how the architecture lever composes with the rest.
