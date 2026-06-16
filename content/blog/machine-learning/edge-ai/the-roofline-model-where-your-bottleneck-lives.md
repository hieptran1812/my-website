---
title: "The roofline model: finding where your edge model's bottleneck actually lives"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Learn to read the single chart that tells you, before you waste a week, whether your edge model is starved for memory bandwidth or for compute — and therefore which optimization lever will actually make it faster."
tags:
  [
    "edge-ai",
    "model-optimization",
    "roofline",
    "arithmetic-intensity",
    "memory-bound",
    "inference",
    "efficient-ml",
    "profiling",
    "quantization",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-1.png"
---

I once spent the better part of a week hand-tuning a matrix-multiply kernel for a depthwise-separable backbone on a Jetson. I rewrote the inner loop, got the tiling right, and doubled the achieved FLOP/s on a microbenchmark. I was thrilled. Then I dropped it into the model, ran the end-to-end latency harness, and the model was exactly as slow as before. Not a little slower than I hoped — *exactly* the same, to within measurement noise.

The reason was humiliating in hindsight and is the single most important idea in this entire series: that layer was never compute-bound. It was **memory-bound**. The hardware was not waiting on the multiply-accumulate units; it was waiting on DRAM to deliver the data. Doubling the math throughput of a unit that is idle three-quarters of the time does nothing. I had spent a week making the engine more powerful in a car that was out of gas.

The tool that would have told me this in fifteen minutes — before I wrote a single line of kernel code — is the **roofline model**. It is a one-page chart that, for a given piece of hardware, draws two ceilings on attainable performance: a slanted one set by memory bandwidth and a flat one set by peak compute. You compute one number for your kernel — its *arithmetic intensity*, the ratio of math done to bytes moved — and that number places the kernel under one ceiling or the other. Whichever ceiling you are sitting under is your bottleneck, and the bottleneck dictates which optimization lever is worth pulling.

![A two-by-two matrix showing the memory roof and compute roof, the ridge point that separates them, which kernels fall on each side, and the corresponding optimization lever to pull.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-1.png)

By the end of this post you will be able to look at any layer — a GEMM, a depthwise convolution, an activation, a single LLM decode step — and say, *before you optimize anything*, whether it is starved for bandwidth or for compute, how much headroom you have, and which of the four levers (quantization, pruning, distillation, efficient architecture) will move the needle. This is the analytical bridge to the rest of the series: every other post is about a lever, and the roofline is how you decide which lever to reach for. If you have not yet read [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), skim it first for the four-lever framing; this post is the diagnostic that sits in front of all four.

## What the roofline actually models

Before any math, fix the mental picture. A processor does two things in a loop: it moves data (between DRAM and the on-chip compute units) and it does arithmetic on that data. These two activities happen at finite, fixed rates. Call them:

- **Peak compute**, $P_{\text{peak}}$, in floating-point operations per second (FLOP/s). For an int8 NPU you would write it in integer operations per second (OP/s); the algebra is identical.
- **Peak memory bandwidth**, $B_{\text{peak}}$, in bytes per second — how fast the chip can stream bytes from DRAM into the compute units.

A kernel needs both. It needs to move some number of bytes and it needs to do some number of FLOPs. The two activities can overlap — modern hardware prefetches the next tile of data while crunching the current one — but they cannot make each other go faster. So the time to run the kernel is *at best* the larger of the two:

$$
t \;\ge\; \max\!\left(\frac{\text{FLOPs}}{P_{\text{peak}}},\; \frac{\text{Bytes}}{B_{\text{peak}}}\right).
$$

If the math takes longer than the data movement, you are compute-bound and the second term is hidden behind the first. If moving the data takes longer than the math, you are memory-bound and the first term is hidden behind the second. The roofline is just this inequality, rearranged into a form you can plot and read at a glance.

The reason this matters so much on the edge is that edge silicon is *bandwidth-poor relative to its compute*. A modern phone NPU might advertise 30–40 TOPS of int8 compute but sit behind LPDDR5 delivering maybe 50–70 GB/s. The ratio of compute-to-bandwidth is enormous, which pushes the break-even point — the *ridge point* we are about to derive — far to the right. That means more of your kernels land on the memory-bound side than your cloud-trained intuition expects. The roofline is the tool that makes this concrete instead of folklore.

There is a deeper, physical reason this imbalance exists and keeps getting worse, and it is worth knowing because it tells you the bandwidth wall is not going away. Transistor density has scaled far faster than the off-chip pin bandwidth and DRAM speed that feed those transistors. Adding more multiply-accumulate units to a chip is comparatively cheap — they are small and dense. Adding off-chip memory bandwidth means more pins, wider buses, faster DRAM, and a great deal more energy and board area, all of which scale slowly and expensively. The industry name for this is the "memory wall," and on a power- and area-constrained edge device it is brutal: you can afford a lot of TOPS but only a little bandwidth. The roofline is the model that turns that physical fact into a per-kernel prediction. When you see the ridge point sitting at a few hundred FLOP/byte, you are looking at the memory wall expressed as a number.

A second physical fact that the roofline encodes implicitly is *energy*. Moving a byte from DRAM costs on the order of a hundred times more energy than the floating-point operation that consumes it — a 32-bit DRAM access is roughly 640 picojoules on a typical mobile process, while a 32-bit floating-point multiply-add is under 4 picojoules. So a memory-bound kernel is not just slow; it is the dominant *energy* sink in your model, which on a battery-powered device matters as much as latency. Cutting bytes moved (the memory lever) therefore wins twice: it speeds up the bandwidth-bound kernel *and* it cuts the energy bill. This is the unifying reason quantization is the highest-leverage edge optimization — fewer bytes is simultaneously less time and less energy on the most expensive operation the chip performs.

## The science: deriving arithmetic intensity and the roofline equation

### Arithmetic intensity, defined carefully

The pivotal quantity is **arithmetic intensity**, written $I$. It is the number of FLOPs a kernel performs divided by the number of bytes it moves through the memory system:

$$
I \;=\; \frac{\text{FLOPs}}{\text{Bytes moved}} \quad\left[\frac{\text{FLOP}}{\text{byte}}\right].
$$

The units are FLOP per byte. A kernel with high intensity does a lot of arithmetic for every byte it reads or writes; a kernel with low intensity does almost no arithmetic per byte and spends its life waiting on the memory bus.

The subtle and crucial word is *which* bytes. Arithmetic intensity is defined with respect to a specific level of the memory hierarchy, and the answer changes depending on which level you mean. The classic Williams–Waterman–Patterson roofline counts **DRAM traffic** — bytes that cross the boundary between off-chip main memory and the on-chip caches. This is the right denominator for the question "is my kernel bound by main-memory bandwidth?", which is almost always the binding question on edge devices, because off-chip LPDDR is the slowest tier and the one that costs the most energy per byte. We will return later to the fact that you can also draw rooflines for cache or scratchpad traffic, and that doing so refines the picture; but for the headline model, **bytes means DRAM bytes moved**.

Be precise about what counts as a FLOP, too. A multiply-accumulate (MAC) is conventionally counted as two FLOPs — one multiply and one add. So a matrix multiply $C = AB$ with $A$ of shape $m \times k$ and $B$ of shape $k \times n$ does $m \cdot n \cdot k$ MACs, which is $2mnk$ FLOPs. Get this factor-of-two convention straight, because it shifts every intensity number by a factor of two and you will confuse yourself comparing against a vendor's published peak (vendors usually quote MAC-based TOPS, where each MAC is one "operation," so check whether their peak is in FLOPs or MACs before you divide).

It also matters which bytes you count on the *write* side and whether they are counted once. The classic roofline counts every byte that crosses the DRAM boundary in either direction — reads of weights and input activations, and writes of output activations. If a kernel reads a tensor, then a later kernel reads it again, that is two trips and two contributions to the byte total. This is why a model expressed as many small unfused kernels moves far more bytes than the same computation fused: each unfused intermediate is written to DRAM by one kernel and read back by the next, doubling its contribution. The roofline's denominator is *traffic*, not *unique data* — and traffic is exactly what fusion reduces. Keeping this straight is the difference between an analytical intensity that matches the hardware counter and one that is optimistically too high because you assumed perfect reuse that the real kernel never achieves.

One more definitional subtlety, because it bites people moving from fp32 cloud training to int8 edge inference: the "FLOP" in arithmetic intensity should match the precision your *peak* roof is quoted in. If your accelerator's flat roof is 40 TOPS of int8, then count integer MACs against int8 byte sizes; if it is 10 TFLOP/s of fp16, count fp16 FLOPs against 2-byte operands. Mixing an int8 byte count with an fp16 peak roof — or vice versa — silently shifts your kernel to the wrong side of the ridge. The cleanest discipline is to pick the precision you will actually deploy in, draw the roof for *that* precision, and compute intensity in the same units throughout.

### The roofline equation and the ridge point

Now turn the time inequality into a performance ceiling. Define attainable performance as FLOPs divided by time, in FLOP/s. Take the best case ($t$ equal to the max term) and substitute:

$$
P_{\text{attainable}} \;=\; \frac{\text{FLOPs}}{t} \;=\; \frac{\text{FLOPs}}{\max\!\left(\dfrac{\text{FLOPs}}{P_{\text{peak}}},\; \dfrac{\text{Bytes}}{B_{\text{peak}}}\right)}.
$$

Divide numerator and denominator through, and using $I = \text{FLOPs} / \text{Bytes}$ you arrive at the canonical roofline equation:

$$
\boxed{\,P_{\text{attainable}}(I) \;=\; \min\!\big(P_{\text{peak}},\; I \cdot B_{\text{peak}}\big)\,}
$$

Read it slowly. Attainable performance is the *minimum* of two terms. The first, $P_{\text{peak}}$, is a constant — it does not depend on the kernel at all. On a plot of performance versus intensity it is a **flat horizontal line**: the compute roof. The second, $I \cdot B_{\text{peak}}$, grows linearly with the kernel's intensity — more arithmetic per byte means the same bandwidth buys you more FLOP/s. On a log-log plot this is a **straight slanted line of slope one**: the memory roof, whose height is set entirely by the bandwidth $B_{\text{peak}}$.

The two roofs cross. Set them equal and solve for the intensity at which the bandwidth-limited performance just reaches peak compute:

$$
P_{\text{peak}} \;=\; I^{*} \cdot B_{\text{peak}} \quad\Longrightarrow\quad I^{*} \;=\; \frac{P_{\text{peak}}}{B_{\text{peak}}}.
$$

This $I^{*}$ is the **ridge point**. It is the arithmetic intensity at which a kernel exactly balances the chip's compute and bandwidth. It is a property of the *hardware*, not of any kernel. To the left of the ridge ($I < I^{*}$), the slanted roof is lower, so the kernel is **memory-bound**: it cannot get enough bytes to keep the math units busy. To the right ($I > I^{*}$), the flat roof is lower, so the kernel is **compute-bound**: the bytes arrive faster than the math can consume them.

The figure in the introduction lays this out as the two roofs, the ridge point, and which kernels fall on which side. The whole diagnostic reduces to one comparison: **is my kernel's $I$ bigger or smaller than the chip's $I^{*}$?**

#### Worked example: where is the ridge point on real edge hardware?

Take three concrete targets and compute $I^{*}$ for each.

- **Jetson Orin Nano (8 GB).** The advertised peak is roughly 40 TOPS of int8, or about 20 TFLOP/s of fp16 on the GPU. The LPDDR5 memory delivers about 68 GB/s. Using fp16: $I^{*} = 20\times10^{12} \,/\, 68\times10^{9} \approx 294$ FLOP/byte. Even at the lower int8 number against the same bandwidth the ridge sits in the hundreds of OP/byte.
- **Raspberry Pi 5 CPU.** Four Cortex-A76 cores at 2.4 GHz with NEON give on the order of 30–40 GFLOP/s of fp32 in practice; the LPDDR4X bandwidth is about 17 GB/s shared. $I^{*} \approx 35\times10^{9} \,/\, 17\times10^{9} \approx 2$ FLOP/byte — much lower, because this chip is bandwidth-rich relative to its modest compute.
- **An NPU-class SoC** with 30 TOPS int8 behind 60 GB/s LPDDR5: $I^{*} = 30\times10^{12} \,/\, 60\times10^{9} = 500$ OP/byte.

Sit with that spread. On the Jetson GPU and the NPU, your kernel needs to do *hundreds* of FLOPs per byte before it stops being memory-bound. Almost nothing in a depthwise-separable network or an LLM decode loop clears that bar. On the Pi CPU the ridge is at $I \approx 2$, so the same kernel might be compute-bound there. **The same model can be memory-bound on one device and compute-bound on another.** This is why a roofline is per-(model, hardware) and never a property of the model alone.

## Compute-bound versus memory-bound, made physical

The abstraction is the inequality; the physical reality is what the silicon does in the two regimes.

![A before-and-after comparison contrasting a compute-bound kernel with busy MAC units against a memory-bound kernel with idle MAC units waiting on DRAM.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-2.png)

In the **compute-bound** regime, the data arrives at the math units at least as fast as they can chew it. The MAC array runs near 100% utilization. The bottleneck is the number of MAC units and their clock. Here, and only here, does the folklore optimization "make the math faster" pay off: a 2× faster MAC array gives roughly a 2× faster kernel. This is the regime where TOPS marketing numbers actually predict speed.

In the **memory-bound** regime, the math units finish each tile and then sit idle waiting for the next batch of bytes to arrive over the bus. Utilization can be under 15%. Doubling the MAC throughput changes nothing because the MACs were never the constraint — they were already waiting. The only thing that helps is moving fewer bytes (so each idle-wait is shorter) or moving them faster (higher bandwidth, which you usually cannot change in firmware). This is the regime I was stuck in on the Jetson, and it is the default regime for on-device LLMs and for the lightweight convolutions that dominate mobile vision backbones.

The figure above contrasts the two regimes side by side: same wall-clock budget, radically different use of it. Internalize that a memory-bound kernel with idle MACs is the *normal* case on edge accelerators, not a pathology.

## Working out the intensity of real kernels

Let us derive $I$ for the kernels that actually show up in edge models. This is where the model stops being a chart and starts being a diagnostic you can run in your head.

### A GEMM — usually compute-bound

A general matrix multiply $C = AB$ with $A \in \mathbb{R}^{m\times k}$, $B \in \mathbb{R}^{k\times n}$ does $2mnk$ FLOPs. The data it must move, in the ideal case where each input and output is touched once in fp32 (4 bytes), is

$$
\text{Bytes} = 4\,(mk + kn + mn),
$$

reading both inputs and writing the output. So

$$
I_{\text{GEMM}} = \frac{2mnk}{4\,(mk + kn + mn)}.
$$

For a square problem $m=n=k=N$, this simplifies to $I = \frac{2N^3}{4\cdot 3N^2} = \frac{N}{6}$. The intensity grows *linearly with the matrix dimension*. A $1024\times1024$ GEMM has $I \approx 170$ FLOP/byte — comfortably right of the Jetson ridge, so compute-bound. A tiny $64\times64$ GEMM has $I \approx 11$ — memory-bound on the same chip. The very same operation flips sides of the ridge depending on size. This is why big batched matmuls are the one thing edge accelerators run near peak, and why making your matmuls *bigger* (by batching, by fusing) is itself an optimization: it raises $I$.

The caveat is that the formula above assumes perfect reuse — each element loaded once. In reality a naive GEMM re-reads $B$ for every row of $A$, blowing the byte count up and crushing $I$. Achieving the $N/6$ intensity is exactly what cache blocking and tiling are *for*: they keep tiles of $A$ and $B$ resident in SRAM so the DRAM byte count stays near the ideal. Blocking does not change the FLOPs; it changes the denominator. We will see this again when we discuss the hierarchical roofline.

It is worth seeing how badly reuse can fail, because it explains a common confusion. A textbook triple-loop GEMM with no blocking streams the entire $B$ matrix from DRAM once per row of $A$ — that is $m$ full reads of $B$, so bytes balloon to $\approx 4(mk + m\cdot kn + mn)$ and intensity collapses toward $2mnk / (4mkn) = 1/2$ FLOP/byte, *independent of $N$*. So an unblocked GEMM is memory-bound no matter how large you make it, while a blocked GEMM of the same size is compute-bound. Same FLOPs, same result, $300\times$ difference in DRAM traffic — and the roofline shows the two implementations as two points at wildly different intensities, with the blocked one snapped up against the compute roof and the naive one stranded on the memory roof. When a vendor BLAS hits 80% of peak and your hand-rolled GEMM hits 5%, this is almost always why: not slow math, but a byte count $10$–$100\times$ too high from missing reuse. The roofline turns "my GEMM is slow" into the precise diagnosis "my GEMM's measured intensity is far below its ideal, so it is moving too many bytes — fix the blocking, do not touch the math."

### A depthwise convolution — low intensity, memory-bound

Depthwise convolution is the efficiency trick at the heart of MobileNet and its descendants: instead of one filter mixing across all input channels, each channel gets its own small spatial filter. For an input of $H \times W \times C$ with a $k\times k$ depthwise filter, the FLOP count is

$$
\text{FLOPs} = 2 \cdot H \cdot W \cdot C \cdot k^2,
$$

because each of the $HWC$ output elements is a $k^2$ MAC reduction. That is *tiny* — there is no $C^2$ term because there is no cross-channel mixing. But the bytes moved are dominated by reading the $HWC$ activations and writing the $HWC$ outputs:

$$
\text{Bytes} \approx 4\,(2 \cdot HWC + C k^2),
$$

with the weight term $Ck^2$ negligible. So the intensity is roughly

$$
I_{\text{dwconv}} \approx \frac{2 HWC k^2}{4 \cdot 2 HWC} = \frac{k^2}{4}.
$$

For a $3\times3$ depthwise kernel, $I \approx 9/4 \approx 2.25$ FLOP/byte — *independent of the feature-map size*, and pinned to the low single digits. On any edge accelerator with $I^{*}$ in the hundreds, a depthwise conv is dramatically memory-bound. It does almost no math; it is entirely about shuttling the activation tensor in and out of DRAM.

This is the dirty secret of "efficient" architectures, and it is exactly the observation the ShuffleNetV2 authors made: **FLOPs are not latency.** MobileNet slashes FLOPs by replacing dense convolutions with depthwise-separable ones, but the depthwise stage has such low arithmetic intensity that on real hardware it can be *bandwidth-bound and therefore slow per FLOP*. A model can have one-ninth the FLOPs of a ResNet and not run one-ninth as fast, because the FLOPs it kept are the low-intensity kind. The roofline is the precise statement of why. (We dig into this trade-off in the architecture posts; here, just note that the roofline is the lens that exposes it.)

### A pointwise (1×1) convolution — intensity rides on the channel count

The other half of a depthwise-separable block is the pointwise convolution: a $1\times1$ conv that mixes across channels, from $C_{\text{in}}$ to $C_{\text{out}}$. Its FLOPs are $2 \cdot H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}}$ — there *is* a product of channel counts here, because every output channel is a full reduction over input channels. The bytes are the activations in and out plus the weights $C_{\text{in}} C_{\text{out}}$:

$$
I_{1\times1} \approx \frac{2 H W C_{\text{in}} C_{\text{out}}}{4\,\big(H W C_{\text{in}} + H W C_{\text{out}} + C_{\text{in}} C_{\text{out}}\big)}.
$$

When the spatial map is large relative to the channel count ($HW \gg C$), the activation terms dominate the denominator and $I \approx \tfrac{1}{2} \tfrac{C_{\text{in}} C_{\text{out}}}{C_{\text{in}} + C_{\text{out}}} \approx C/4$ for $C_{\text{in}}\approx C_{\text{out}} = C$. So a pointwise conv's intensity scales with the channel width: at $C = 32$ it is around 8 FLOP/byte (memory-bound on an edge accelerator), at $C = 512$ it is around 128 FLOP/byte (approaching compute-bound). This is why early, wide-spatial / narrow-channel stages of a mobile network are memory-bound and late, narrow-spatial / wide-channel stages drift toward compute-bound — and it is why a single model can straddle the ridge, with different layers wanting different levers. The roofline applied *per layer* is what reveals that the right optimization is not uniform across the network.

The composition lesson is sharp: a depthwise-separable block pairs a *memory-bound* depthwise conv ($I \approx 2.25$) with a pointwise conv whose intensity depends on width. The block as a whole is dragged toward memory-bound by the depthwise stage and by the intermediate activation that, unfused, round-trips to DRAM between the two. Fusing the depthwise → pointwise → activation sequence into one kernel — keeping the intermediate on-chip — is therefore the single highest-value transformation for these blocks, and it is a *byte* reduction, exactly what the roofline prescribes for the memory-bound regime.

### An elementwise / activation op — always memory-bound

A pointwise activation like ReLU, GELU, a layernorm, or a residual add does on the order of one to a few FLOPs per element while reading and writing the entire tensor. For an $N$-element fp32 tensor it reads $4N$ bytes and writes $4N$ bytes for roughly $cN$ FLOPs with $c$ a small constant:

$$
I_{\text{act}} \approx \frac{cN}{8N} = \frac{c}{8} \ll 1 \ \text{FLOP/byte}.
$$

Activation ops have arithmetic intensity well below one. They are *always* memory-bound, on every device, with no exception, because there is essentially no math to hide the memory traffic behind. This is the single strongest argument for **operator fusion**: if you fuse the activation into the preceding matmul or convolution, the intermediate tensor never has to round-trip to DRAM. You eliminate the $8N$ bytes entirely. Fusion does not reduce FLOPs at all — it reduces bytes, which is precisely the lever a memory-bound op needs. The roofline tells you fusion will help an activation op and tells you it will do *nothing* for a compute-bound GEMM.

### LLM prefill versus decode — the central fact of on-device LLMs

Here is the one that matters most for anyone shipping a language model to a phone or a laptop, and it is worth deriving carefully because it explains the entire shape of on-device LLM performance.

![A before-and-after comparison showing LLM prefill as compute-bound with each weight reused across all prompt tokens, versus decode as memory-bound with every weight re-streamed per token.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-4.png)

A transformer's forward pass is dominated by large matmuls: the attention projections and the feed-forward layers. Consider a model with $P$ parameters (weights), stored at $b$ bytes each. Per token, the matmuls do roughly $2P$ FLOPs — the standard "two FLOPs per parameter per token" rule for a forward pass. To do that, the hardware must read all $P$ weights from memory, which is $bP$ bytes.

**Prefill.** When you feed the model a prompt of $S$ tokens, it processes all $S$ at once. The weights are loaded *once* and reused across all $S$ tokens. So:

$$
I_{\text{prefill}} \approx \frac{2 P \cdot S}{b P} = \frac{2 S}{b}.
$$

With $S = 2048$ tokens and $b = 2$ bytes (fp16), $I_{\text{prefill}} \approx 2048$ FLOP/byte — far to the right of any ridge. **Prefill is compute-bound.** The matmuls are tall and dense; the weight reads amortize over thousands of tokens. This is why your prompt-processing throughput can be high (hundreds to thousands of tokens per second) and scales with compute.

**Decode.** After the prompt, generation is autoregressive: the model emits one token, appends it, and runs the forward pass again for the *next single token*. Now $S = 1$. Each weight is read from memory and used for exactly *one* token's worth of math before being discarded:

$$
I_{\text{decode}} \approx \frac{2 P \cdot 1}{b P} = \frac{2}{b}.
$$

With $b = 2$ bytes, $I_{\text{decode}} = 1$ FLOP/byte. With int8 ($b=1$), it is 2 FLOP/byte. With 4-bit weights ($b = 0.5$), it is 4 FLOP/byte. These are *minuscule* — far to the left of any edge ridge point. **Single-token decode is profoundly memory-bound.** The arithmetic per token is trivial; the cost is dominated by streaming the entire weight matrix out of DRAM, once per token.

This single fact explains the whole feel of running an LLM on-device. It is why a 7B model at fp16 (14 GB of weights) generating one token at a time on a chip with 70 GB/s bandwidth has a *hard floor* on its per-token latency:

$$
t_{\text{token}} \ge \frac{\text{bytes per token}}{B_{\text{peak}}} = \frac{14\times10^{9}}{70\times10^{9}} = 0.2\ \text{s} \;\Rightarrow\; \le 5\ \text{tok/s}.
$$

No amount of faster math gets you below that floor, because the floor is set by bandwidth, not compute. It is also exactly why **weight quantization is the killer optimization for on-device LLMs**: cut the bytes per weight from 2 (fp16) to 0.5 (4-bit) and you cut the bytes streamed per token by 4×, which directly lifts the token-rate ceiling by up to 4×. We will quantify this in the worked example below. For the deeper mechanics of decoding and the KV cache — which adds its own memory traffic that grows with context length — see [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) and the broader survey in [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques).

The before-and-after figure above shows the split cleanly: prefill on the compute-bound right, decode on the memory-bound left, the same weights playing two completely different roles.

## A table of operational intensities

Pulling the derivations together gives a layer-by-layer cheat sheet. The numbers below assume fp32 activations unless noted and a single inference (batch 1); they are order-of-magnitude figures meant for placing layers relative to a ridge point in the hundreds, not exact accounting.

![A matrix table listing common edge layers with their FLOPs, bytes moved, arithmetic intensity, and whether each is compute-bound or memory-bound.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-3.png)

| Layer type | FLOPs (rough) | Bytes moved (rough) | $I$ (FLOP/byte) | Bound |
|---|---|---|---|---|
| Large dense GEMM ($N\!\approx\!1024$) | $\sim 2\,\text{GFLOP}$ | $\sim 12\,\text{MB}$ | $\sim 170$ | **Compute** |
| $3\times3$ conv, large channels | $\sim 150\,\text{MFLOP}$ | $\sim 3\,\text{MB}$ | $\sim 25$ | **Compute** |
| Pointwise ($1\times1$) conv | $2HWC_{\text{in}}C_{\text{out}}$ | $\sim HW(C_{\text{in}}{+}C_{\text{out}})$ | $\sim C/2$ | Depends on $C$ |
| **Depthwise** $3\times3$ conv | $2HWCk^2$ | $\sim 8HWC$ | $\sim 2.25$ | **Memory** |
| Activation / norm / residual | $\sim cN$ | $\sim 8N$ | $\sim 0.5$ | **Memory (always)** |
| LLM prefill ($S{=}2048$, fp16) | $2PS$ | $bP$ | $\sim 2000$ | **Compute** |
| LLM decode ($S{=}1$, fp16) | $2P$ | $bP$ | $\sim 1$ | **Memory (deeply)** |
| LLM decode ($S{=}1$, 4-bit) | $2P$ | $0.5P$ | $\sim 4$ | **Memory** |

The matrix figure above renders the same five-row comparison; the takeaway is the bimodal split. A handful of layers (big GEMMs, large dense convs, prefill) sit high and compute-bound; everything lightweight (depthwise, pointwise at small channel counts, activations, decode) crowds the low-intensity, memory-bound end. On an edge accelerator with a ridge in the hundreds, **the majority of a typical mobile model's runtime is memory-bound**, which is the opposite of what people assume after counting FLOPs. This is the empirical heart of why the four-lever framing in [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) leads with quantization for edge work.

## Why the roofline picks your optimization lever

This is the payoff — the reason this is the keystone post. The roofline does not just diagnose; it *prescribes*.

![A decision tree that routes a memory-bound kernel toward quantization and fusion and a compute-bound kernel toward pruning, distillation, or faster MAC units.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-5.png)

If a kernel is **memory-bound** (left of the ridge), the constraint is bytes. The levers that help are the ones that reduce bytes moved:

- **Quantization** is the first and biggest. Going from fp32 to int8 cuts bytes-per-weight by 4×; from fp16 to 4-bit by 4×. For a memory-bound kernel this translates almost directly into speed, because the runtime *is* the byte count divided by bandwidth. And here is the beautiful corollary the roofline makes obvious: **when you are memory-bound, extra FLOPs are free.** A dequantize-then-multiply has more arithmetic than a plain fp32 multiply, but the math units were idle anyway, so the added FLOPs hide entirely behind the (now smaller) memory traffic. This is why int8 matmuls that *look* more expensive per element can run faster — they move 4× fewer bytes through the bottleneck.
- **Operator fusion** raises $I$ by keeping intermediates on-chip, eliminating round-trips to DRAM.
- **Bigger tiles / batching** raises $I$ by reusing each loaded byte for more math, sliding the kernel rightward toward the ridge.

If a kernel is **compute-bound** (right of the ridge), the constraint is FLOPs. Reducing bytes does nothing — you already have data to spare. The levers that help are the ones that reduce FLOPs or speed up MACs:

- **Pruning** (especially structured / 2:4 sparsity that hardware can exploit) removes MACs.
- **Distillation** to a smaller architecture removes MACs.
- **Faster MAC units** — moving the kernel onto tensor cores or an NPU, or using a lower-precision MAC the hardware runs at higher throughput — raises the flat roof itself.

The decision tree figure encodes exactly this routing. The discipline it enforces is what saves the wasted week: **you do not pick a lever from habit; you measure $I$, compare it to $I^{*}$, and let the binding roof choose.** Quantization on a compute-bound kernel that already has plenty of bandwidth headroom buys you almost nothing (it might even hurt if the int8 path is less mature than the fp16 one). Kernel-tuning a memory-bound kernel buys you nothing — that was my Jetson mistake. The roofline is the map that tells you which half of the problem you are actually in.

This is exactly where the roofline plugs into the four-lever frame that organizes the whole series. The four levers — quantization, pruning/sparsity, distillation, and efficient-architecture/NAS — are not interchangeable; each attacks a different term in the roofline. Quantization attacks *bytes per weight* (the denominator of $I$, and the memory roof you sit on). Pruning and distillation attack *FLOPs* (the numerator, and where you sit relative to the compute roof). Efficient architectures attack *both at once* but, as the depthwise story showed, can accidentally trade compute-bound FLOPs for memory-bound ones and lose on real hardware. And the compiler/runtime layer — fusion, tiling, the delegate that maps your graph onto the NPU — attacks *traffic and the memory level you stream from*. So the roofline is not a sibling of the four levers; it is the *diagnostic that sits in front of them*, telling you which lever's term is the binding one for this kernel on this chip. Read top-down: profile → place on roofline → identify binding roof → pull the lever that moves that roof. The capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) runs this exact loop end to end on a real model.

The stress test that catches the nastiest edge surprise: **what if the NPU does not support your op and it falls back to CPU?** Suppose your roofline says a layer is compute-bound and you correctly quantize the whole model to int8 to run on the NPU. But the NPU runtime does not implement, say, a particular activation or a fancy attention variant, so that op silently falls back to the CPU. Now every time control reaches that op, the runtime copies the activation tensor *off the NPU to CPU memory and back* — pure DRAM traffic with zero useful FLOPs, the most memory-bound thing imaginable. Your beautiful compute-bound roofline analysis is now dominated by an op the model could not even see, and end-to-end latency is far worse than predicted. The lesson is that the per-kernel roofline must be paired with a *coverage* check: profile the end-to-end graph and confirm every op actually ran on the intended accelerator. An unsupported op is not a roofline problem; it is a roofline *trap*, and the only way to find it is the end-to-end trace, not the per-kernel arithmetic.

### Quantization slides a memory-bound kernel up the roof

Let us make the "quantization helps memory-bound kernels" claim quantitative, because it is the single most actionable consequence of the model.

![A before-and-after figure showing an fp32 kernel low on the bandwidth roof and the same kernel after int8 quantization climbing the roof with four times fewer bytes and a near four times speedup.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-6.png)

Take a memory-bound kernel at fp32 with intensity $I_0$ and runtime $t_0 = \text{Bytes}_0 / B_{\text{peak}}$ (the memory term dominates, by assumption). Quantize the weights to int8. The FLOPs are unchanged, but the bytes moved drop by 4× (4 bytes → 1 byte per weight, assuming weights dominate the traffic). So:

- New intensity: $I_1 = \text{FLOPs} / (\text{Bytes}_0 / 4) = 4 I_0$. The kernel moves *up and to the right* along the slanted roof.
- New runtime, if still memory-bound: $t_1 = (\text{Bytes}_0 / 4) / B_{\text{peak}} = t_0 / 4$.

So you get up to a **4× speedup**, capped at the moment the kernel crosses the ridge and becomes compute-bound (after which further byte reductions stop helping and you would switch levers). The figure above shows the climb: fp32 low on the slope, int8 four times higher in intensity and roughly four times faster, both still under the slanted roof. This is the geometric picture behind every "int8 made our model 3–4× faster" result, and the roofline tells you the *ceiling* of that speedup before you run the conversion. For how int8 and 4-bit quantization actually encode the weights — the GGUF k-quant types, the scale/zero-point math — see [how quantization works: GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded).

## Building a roofline for your model on your hardware

Theory is cheap. Here is the practical workflow to build a real roofline for *your* model on *your* device, end to end. It is a five-step measurement loop, and every step is something you can run today.

![A timeline of the five-step workflow to build a roofline: measure the two hardware ceilings, profile per-kernel FLOPs and bytes, compute intensity, plot against the roofs, and pick the lever.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-8.png)

### Step 1 — pin the two roofs (measure peak FLOP/s and peak bandwidth)

You need two numbers for your hardware: $P_{\text{peak}}$ and $B_{\text{peak}}$. Start with the vendor datasheet, but *verify with microbenchmarks*, because achievable peak is usually 60–85% of the marketing number.

For peak bandwidth, the standard tool is the STREAM benchmark (a trivial Triad kernel `a[i] = b[i] + s*c[i]` that is purely bandwidth-bound). On a GPU you can write a tiny copy kernel; on a CPU, the `STREAM` C benchmark is canonical. For peak compute, run a large, well-tiled GEMM (e.g. a $4096\times4096$ matmul through cuBLAS, oneDNN, or your accelerator's BLAS) and divide its FLOPs by its measured time.

```python
import torch, time

def measure_peak_flops(n=4096, dtype=torch.float16, device="cuda", iters=50):
    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)
    # warm-up: trigger autotuning and clocks ramping up
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2 * n**3            # 2*m*n*k for a square GEMM
    return flops / dt           # achieved FLOP/s

def measure_peak_bw(n=1 << 26, dtype=torch.float16, device="cuda", iters=50):
    # STREAM-style copy: reads n elems, writes n elems => 2 * n * bytes moved
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.empty_like(x)
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    bytes_moved = 2 * n * x.element_size()
    return bytes_moved / dt     # achieved bytes/s

P_peak = measure_peak_flops()
B_peak = measure_peak_bw()
print(f"Peak compute  ~ {P_peak/1e12:.1f} TFLOP/s")
print(f"Peak bandwidth ~ {B_peak/1e9:.0f} GB/s")
print(f"Ridge point I* = {P_peak / B_peak:.0f} FLOP/byte")
```

Those three printed numbers — peak compute, peak bandwidth, ridge point — *are* your roofline. Everything else is placing kernels on it.

### Step 2 — profile per-kernel FLOPs and bytes moved

Now you need, for each kernel in your model, its FLOPs and its DRAM bytes. There are three tiers of tooling, from rough to exact.

**Rough, analytical (good enough to triage):** compute FLOPs and bytes from the layer shapes, with the formulas we derived. Libraries like `fvcore`, `ptflops`, `thop`, or `torch.utils.flop_counter.FlopCounterMode` count FLOPs automatically:

```python
import torch
from torch.utils.flop_counter import FlopCounterMode

model = build_model().eval().cuda()
x = torch.randn(1, 3, 224, 224, device="cuda")

with FlopCounterMode(model, display=False) as fcm:
    model(x)

# per-module FLOP counts (in FLOPs, 1 MAC = 2 FLOPs by torch's convention)
flops_by_module = fcm.get_flop_counts()
for mod, counts in flops_by_module.items():
    total = sum(counts.values())
    if total:
        print(f"{mod:40s} {total/1e6:8.1f} MFLOP")
```

For bytes you still need the activation and weight shapes; a quick estimate is `bytes = (weight_numel + activation_in_numel + activation_out_numel) * dtype_size`.

**Medium, runtime-measured (`torch.profiler`):** the PyTorch profiler records per-op CPU/CUDA time and, with `profile_memory=True` and `with_flops=True`, FLOP estimates and memory activity. Reading the trace tells you which ops dominate wall time — often a surprise.

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    with record_function("model_inference"):
        for _ in range(20):
            model(x)
    torch.cuda.synchronize()

# sort by GPU time to find the kernels that actually cost you
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or Perfetto
```

The `with_flops=True` column gives FLOPs; `profile_memory=True` gives allocation activity. The trace's per-op CUDA time, divided into the analytical byte and FLOP counts, lets you place each op on the roofline.

**Exact, hardware-measured (Nsight Compute / `ncu`):** for the ground truth on NVIDIA hardware, NVIDIA Nsight Compute reads the actual hardware performance counters — DRAM bytes transferred, FLOPs executed, and the achieved performance — and it has a built-in roofline analysis section that draws your kernel on the chart for you. Run:

```bash
# Profile one kernel and emit the roofline + memory-workload sections.
ncu --set roofline \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    -o my_roofline_report \
    python infer.py

# Or target a specific kernel by name and limit launches:
ncu --kernel-name regex:"gemm|conv" --launch-count 5 \
    --set roofline python infer.py
```

Open the `.ncu-rep` in the Nsight Compute UI and the "GPU Speed of Light Roofline Chart" places every profiled kernel against the measured memory and compute roofs — the dots-on-the-roofline picture, drawn from real counters rather than your arithmetic. The DRAM bytes come straight from the `dram__bytes` metric, so you are reading the true denominator of $I$, not estimating it. On Intel CPUs/GPUs, **Intel Advisor** has the equivalent automated roofline feature; on AMD, `rocprof` plus the roofline scripts.

### Step 3 — compute $I$ and classify each kernel

With FLOPs and bytes per kernel and your $P_{\text{peak}}$, $B_{\text{peak}}$ in hand, the classification is three lines:

```python
def classify_kernel(flops, bytes_moved, P_peak, B_peak):
    I = flops / bytes_moved                  # arithmetic intensity, FLOP/byte
    I_star = P_peak / B_peak                 # hardware ridge point
    roof = min(P_peak, I * B_peak)           # attainable FLOP/s on this chip
    bound = "compute-bound" if I >= I_star else "memory-bound"
    # how close are we to the roof we sit under?
    achieved = flops / measured_time_s       # from the profiler
    utilization = achieved / roof
    return {
        "I": I,
        "I_star": I_star,
        "bound": bound,
        "attainable_flops": roof,
        "utilization_vs_roof": utilization,  # < ~0.7 means there is headroom to chase
    }
```

If `bound == "memory-bound"`, your lever is quantization/fusion. If `bound == "compute-bound"`, your lever is pruning/distillation/faster-MACs. If `utilization_vs_roof` is already near 1.0, the kernel is *optimal for this hardware* and you should stop touching it and go look at the next-biggest contributor to wall time. That last check is the one that would have saved my week: a memory-bound kernel already near its memory roof has *no compute headroom to recover*, so kernel-tuning the math is pointless.

### Step 4 and 5 — plot and decide

Scatter every kernel as a point at $(I, \text{achieved FLOP/s})$ on a log-log plot with the two roofs drawn, then act on the binding roof per the decision tree. A minimal plotter:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_roofline(P_peak, B_peak, kernels):
    I = np.logspace(-2, 4, 200)
    roof = np.minimum(P_peak, I * B_peak)
    plt.loglog(I, roof, 'k-', lw=2, label="roofline")
    I_star = P_peak / B_peak
    plt.axvline(I_star, ls="--", color="gray")
    plt.text(I_star, P_peak * 0.5, "ridge", rotation=90)
    for name, (Ik, achieved) in kernels.items():
        plt.scatter(Ik, achieved)
        plt.annotate(name, (Ik, achieved))
    plt.xlabel("Arithmetic intensity (FLOP/byte)")
    plt.ylabel("Performance (FLOP/s)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.savefig("roofline.png", dpi=150, bbox_inches="tight")
```

Kernels far below the roof they sit under have headroom; kernels hugging the roof are done. Memory-bound kernels with headroom are usually a fusion or tiling problem; memory-bound kernels *on* the roof want fewer bytes (quantize). The timeline figure above is this exact five-step loop laid out left to right.

### Reading bytes-moved out of a profiler trace

The step that trips people up is getting honest *bytes moved*, because the analytical estimate and the hardware reality often disagree. Here is how to read each tool.

In a `torch.profiler` Chrome trace, you do not get DRAM bytes directly, but you get per-op input and output shapes (with `record_shapes=True`) and CUDA memory-copy events. The practical move is to sum, per op, the byte sizes of its input tensors, output tensors, and parameters, and treat that as a lower bound on traffic; then compare the op's measured CUDA time against `bytes / B_peak` and `flops / P_peak`. If measured time tracks the memory term, you have confirmed memory-bound empirically, not just by intensity arithmetic.

In **Nsight Compute**, the truth is in two metrics: `dram__bytes.sum` (total bytes that crossed the DRAM interface for this kernel launch) and the FLOP counters (`sm__sass_thread_inst_executed_op_*` rolled up by the roofline section). Their ratio is the *measured* arithmetic intensity, and it is the one to trust. When the measured `dram__bytes` is much larger than your analytical estimate, the kernel is re-reading data (poor blocking) or moving uncoalesced/strided data — a finding the analytical roofline could never surface. When it is *smaller* than your estimate, the data was already cache-resident and you should be looking at the L2 roofline instead. Reading `dram__bytes` is, more than any formula in this post, how you build a roofline you can defend in a design review.

A worked reading: suppose `ncu` reports a kernel with `dram__bytes.sum = 3.2 MB`, an executed-FLOP count of `6.4 MFLOP`, and a duration of `120 µs`. Measured intensity is $6.4 / 3.2 = 2$ FLOP/byte. Achieved performance is $6.4\times10^{6} / 120\times10^{-6} = 53$ GFLOP/s. Achieved bandwidth is $3.2\times10^{6} / 120\times10^{-6} = 27$ GB/s. If the chip's $B_{\text{peak}}$ is 60 GB/s, you are at 45% of the bandwidth roof — memory-bound (intensity 2 is left of any edge ridge) *with headroom*, which points at coalescing or a fusion opportunity before you reach for quantization. That chain — counters → intensity → which roof → how much headroom → which lever — is the entire method in one paragraph.

## Two worked examples, start to finish

#### Worked example: a depthwise convolution layer on a Jetson

Take a depthwise $3\times3$ stage from a MobileNet-style backbone: input $112\times112\times144$ at fp16 (2 bytes), stride 1, on a Jetson Orin Nano where we measured $P_{\text{peak}} \approx 18$ TFLOP/s fp16 and $B_{\text{peak}} \approx 60$ GB/s, giving $I^{*} \approx 300$ FLOP/byte.

FLOPs: $2 \cdot HWC \cdot k^2 = 2 \cdot 112 \cdot 112 \cdot 144 \cdot 9 \approx 32.5$ MFLOP.

Bytes moved (fp16, weights negligible): read activations $2 \cdot 112\cdot112\cdot144 \approx 3.6$ MB, write outputs another $3.6$ MB, so $\approx 7.2$ MB.

Intensity: $I = 32.5\times10^{6} \,/\, 7.2\times10^{6} \approx 4.5$ FLOP/byte.

That is $4.5 \ll I^{*} = 300$, so this layer is **deeply memory-bound**. The attainable performance is $I \cdot B_{\text{peak}} = 4.5 \cdot 60\,\text{GB/s} = 270$ GFLOP/s — about 1.5% of the chip's 18 TFLOP/s peak. The decision: do **not** hand-tune the MAC inner loop; the MACs are 98% idle. The roofline says reduce bytes. Two moves: (1) **fuse** the depthwise conv with the following pointwise conv and activation so the $112\times112\times144$ intermediate never touches DRAM — this can roughly halve the bytes and double the achievable speed; (2) **quantize to int8** so activations are 1 byte not 2, halving the traffic again and raising $I$ to $\approx 9$. The predicted speedup from both is roughly 3–4×, and crucially it comes entirely from the memory lever, which the roofline identified before any code was written. If I had instead spent the week on a faster int8 MAC kernel — the lever for the *compute*-bound side — I would have moved a point that was already 98% below its roof and changed nothing.

#### Worked example: an LLM decode step, and why Q4 gives a near-linear speedup

Now the headline edge case: a 7B-parameter LLM generating tokens on a laptop with $B_{\text{peak}} \approx 100$ GB/s (a respectable LPDDR5/unified-memory figure) and $P_{\text{peak}} \approx 4$ TFLOP/s of usable fp16 on the integrated GPU, so $I^{*} \approx 40$ FLOP/byte.

**At fp16** ($b = 2$ bytes/weight), weights are $2 \cdot 7\times10^{9} = 14$ GB. Per decode step the model streams all 14 GB and does $\approx 14$ GFLOP ($2P$). Intensity $I = 14\times10^{9} / 14\times10^{9} = 1$ FLOP/byte — far left of $I^{*}=40$, **memory-bound**. The per-token floor:

$$
t_{\text{token}} \ge \frac{14\ \text{GB}}{100\ \text{GB/s}} = 140\ \text{ms} \;\Rightarrow\; \le 7.1\ \text{tok/s}.
$$

The compute term would be $14\times10^{9}/4\times10^{12} = 3.5$ ms — twenty times smaller, completely hidden. Bandwidth owns the latency.

**At 4-bit** ($b = 0.5$ bytes/weight, e.g. a GGUF `Q4_K_M` quantization), weights are $\approx 3.5$ GB. Same $\approx 14$ GFLOP. Intensity $I = 14\times10^{9} / 3.5\times10^{9} = 4$ FLOP/byte — still left of the ridge, still memory-bound, but the byte count dropped 4×:

$$
t_{\text{token}} \ge \frac{3.5\ \text{GB}}{100\ \text{GB/s}} = 35\ \text{ms} \;\Rightarrow\; \le 28.6\ \text{tok/s}.
$$

A **near-4× speedup in single-stream tokens-per-second**, exactly as the memory roof predicts, because we are sliding the kernel up the slanted roof by cutting bytes 4×. This is why `llama.cpp`'s 4-bit k-quants are the default for on-device LLMs:

```bash
# Convert HF weights to GGUF, then quantize to 4-bit (Q4_K_M).
python convert_hf_to_gguf.py ./Llama-2-7b-hf --outfile llama-7b-f16.gguf --outtype f16
./llama-quantize llama-7b-f16.gguf llama-7b-q4_k_m.gguf Q4_K_M

# Run and measure decode tok/s (-n tokens to generate, -ngl layers on GPU).
./llama-cli -m llama-7b-q4_k_m.gguf -p "Explain the roofline model." \
    -n 256 -ngl 99 --no-display-prompt
# The reported "eval time ... tokens per second" is your decode rate.
```

#### Worked example: the same MobileNet, two devices, two verdicts

To drive home that the roofline is per-(model, hardware), take one pointwise conv from the middle of a MobileNetV3 — input $14\times14\times240$, output $14\times14\times240$, $1\times1$ — and place it on two devices.

FLOPs: $2 \cdot 14 \cdot 14 \cdot 240 \cdot 240 \approx 22.6$ MFLOP. Bytes (fp16): activations in $2\cdot14\cdot14\cdot240 \approx 94$ KB, out another 94 KB, weights $2\cdot240\cdot240 \approx 115$ KB, total $\approx 0.30$ MB. Intensity $I = 22.6\times10^{6} / 0.30\times10^{6} \approx 75$ FLOP/byte.

On the **Jetson Orin Nano** ($I^{*}\approx 300$): $75 < 300$, so this layer is **memory-bound** there — the lever is fuse/quantize. On the **Raspberry Pi 5 CPU** ($I^{*}\approx 2$): $75 \gg 2$, so the very same layer is **compute-bound** there — the lever is fewer FLOPs or faster MACs (NEON, int8 dot-product instructions). One layer, two devices, *opposite verdicts and opposite optimizations*. If you ported a Jetson-tuned int8 model to the Pi expecting the same wins, you would be disappointed: on the Pi this layer was never bandwidth-starved, so the byte reduction from int8 buys little, and what you actually want there is to keep the MAC units fed with a well-vectorized int8 kernel. This is the concrete reason "we quantized it and it was faster on device A but not device B" is not a mystery — it is the ridge point moving.

The stress test the roofline predicts: **batching helps throughput but not single-stream latency.** If you process $N$ requests together, the weights are loaded once and reused across all $N$ tokens, so intensity rises to $\approx N$ FLOP/byte and you move toward the ridge — aggregate tokens/s climbs. But any *individual* user's latency is unchanged or worse, because each token still waits for the same per-step weight stream (now shared). On a phone serving one user, batching is irrelevant; the memory floor is the floor. On a server, batching is everything. The roofline tells you which world you are in. This is the same reasoning the inference-serving literature formalizes; the [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) post goes deeper on continuous batching and why it is a throughput, not latency, lever.

## The hierarchical roofline, and the limits of the model

So far we have drawn *one* roofline, against DRAM bandwidth. Real chips have a memory hierarchy — registers, L1/SRAM scratchpad, L2/shared cache, then DRAM — each with its own bandwidth, each faster and smaller than the next.

![A stack of memory hierarchy levels from registers through L1 and L2 to DRAM, each labeled with its bandwidth and noting that each level has its own roofline.](/imgs/blogs/the-roofline-model-where-your-bottleneck-lives-7.png)

The **hierarchical roofline** (developed by the original roofline authors and the LBNL performance group) draws a separate slanted roof for each memory level, because each level has a different $B_{\text{peak}}$ and your kernel moves a different number of bytes against each. A kernel might be DRAM-bound but, once its working set is blocked to fit in L2, become L2-bound and run much faster — the L2 roof is steeper, so the same intensity buys more performance. The stack figure above shows the four tiers and the key idea: **blocking and fusion do not change FLOPs; they change which level of the hierarchy you are streaming from, moving you onto a steeper roof.** This is the rigorous statement of why cache tiling works. When you measure DRAM bytes with `ncu`'s `dram__bytes` and *also* the L2 bytes with `lts__t_bytes`, you can place the kernel on both rooflines and see whether it is DRAM-bound (data spilling to main memory) or already cache-resident.

The reason the hierarchical view matters in practice is that it tells you *whether blocking is even possible*. Notice that arithmetic intensity computed against a faster level is always at least as high as against DRAM, because the faster level moves fewer (or equal) bytes off-chip. If a kernel is DRAM-bound but its intensity *against L2* is high, then there exists a blocking that keeps the working set in L2 and lifts the kernel onto the steep L2 roof — fusion/tiling will work. But if the kernel's intensity is intrinsically low *at every level* (the depthwise conv and the activation op are like this — they genuinely do almost no math per byte no matter how you stage the data), then no amount of blocking saves you, and the only lever left is reducing the bytes themselves through quantization or eliminating the op via fusion-into-a-neighbor. The hierarchical roofline is how you tell those two cases apart: "memory-bound but blockable" versus "memory-bound and fundamentally low-intensity." They demand different responses, and guessing wrong is another way to lose a week.

Now the honesty section, because every model has a domain of validity and using one outside it is how you get fooled.

**The roofline ignores latency.** It is a *bandwidth-and-throughput* model. It assumes infinite parallelism perfectly hides memory latency behind compute. A kernel with too little parallelism — too few threads in flight to cover the hundreds of cycles of DRAM latency — will run far below its roofline even though, by intensity, it "should" hit it. The roofline cannot see this; it cannot tell you that you launched too few warps or that your batch is too small to fill the pipeline.

**It ignores occupancy and overlap quality.** Real hardware only achieves peak compute when the MAC array is fully fed and only achieves peak bandwidth with enough outstanding memory requests. A kernel can be theoretically compute-bound yet run at 40% of peak because occupancy is low or the memory accesses are uncoalesced/strided (so you move far more bytes than the useful data). The roofline says "you could hit this roof," not "you will."

**It ignores per-op launch overhead.** On the edge especially, a model is often a long chain of *tiny* ops. Each kernel launch has fixed overhead (microseconds on a GPU, dispatch cost on an NPU). A graph of 200 micro-ops can be dominated by launch overhead, not by either roof — which is why graph-level fusion and using a compiler that emits fewer, larger kernels (TensorRT, XLA, the ExecuTorch/LiteRT delegates) matters as much as the per-kernel roofline. See [TensorRT end-to-end inference compiler](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler) for how a graph compiler attacks exactly this.

**It assumes you measured the right bytes.** If your "bytes moved" is the analytical ideal but the real kernel re-reads inputs (poor blocking) or moves padding/uncoalesced data, your point will sit *left* of where you think, and the kernel will look memory-bound for the wrong reason. Always prefer the hardware counter (`dram__bytes`) over the analytical estimate when the stakes are high.

So the roofline is a *triage* tool. It is unbeatable at answering "which half of the problem am I in, and is there headroom?" It is silent on "why am I at 40% of the roof I should hit?" — for that you drop into the profiler's occupancy, warp-stall, and memory-coalescing views. Use the roofline to pick the lever; use the deep profiler to land the punch.

## Results: a measured before → after

Theory predicts a 4× speedup when you quantize a memory-bound LLM decode from fp16 to 4-bit. Here is what that looks like measured, and where reality falls short of the model. These are representative figures for a 7B model on a unified-memory laptop class device (an Apple-silicon-style target, ~100 GB/s) running through `llama.cpp` with the GPU backend; treat them as order-of-magnitude, measured-style numbers, not a specific SKU's certified benchmark.

Before the numbers, the measurement discipline, because a roofline comparison is only as honest as the latencies feeding it. **Warm up first:** the first few iterations pay for clock ramp-up, kernel autotuning, and cold caches; discard them and time the steady state. **Watch thermals:** edge devices throttle. A phone or a Jetson under sustained load drops its clocks within tens of seconds, so a number measured in the first second can be 30% optimistic versus the sustained rate the user actually experiences — report the sustained figure and say how long you ran. **Measure at batch 1** if that is the deployment reality (it usually is on a single-user device); batch-8 throughput is a different metric and conflating them is the most common way roofline claims get oversold. **Report a distribution, not a point:** p50 and p99 (or tokens/s mean and its variance), because a memory-bound kernel's latency is tight but op-launch jitter and thermal events fatten the tail. And **fix the bytes you are counting** — weight-only, or weight-plus-KV-cache — and state it, since the KV cache's contribution grows with context and changes the floor.

| Configuration | Bytes/weight | Weights streamed/token | Predicted floor | Measured decode | Model size | Notes |
|---|---|---|---|---|---|---|
| fp16 baseline | 2.0 | 14.0 GB | ≥ 140 ms (≤ 7.1 tok/s) | ~6.4 tok/s | 13.5 GB | memory-bound, near floor |
| int8 (Q8_0) | 1.0 | 7.0 GB | ≥ 70 ms (≤ 14.3 tok/s) | ~12.1 tok/s | 6.7 GB | ~1.9× faster |
| 4-bit (Q4_K_M) | ~0.55 | ~3.9 GB | ≥ 39 ms (≤ 25.6 tok/s) | ~22 tok/s | 3.8 GB | ~3.4× faster |
| 4-bit, batch 8 | ~0.55 | ~3.9 GB (shared) | — | ~84 tok/s aggregate | 3.8 GB | throughput, not latency |

Read the gap between predicted and measured. The 4-bit case predicts up to 25.6 tok/s and we measure ~22 — about **86% of the roofline ceiling**. Where did the other 14% go? Three places, each one a limit of the model we just listed:

1. **The k-quant format is not exactly 0.5 bytes/weight.** `Q4_K_M` carries 4-bit weights plus per-block scales and a small fraction of more-precise weights, landing around 4.4–4.8 effective bits, so the real byte count is ~10% above the ideal 4-bit floor. The roofline assumed the ideal.
2. **Dequantization and the KV-cache traffic add bytes the simple model ignored.** Each step also reads/writes the KV cache, whose size grows with context length; at long context this becomes a non-trivial slice of the per-token bytes and pulls measured tok/s further below the weight-only floor.
3. **Launch overhead and imperfect overlap.** The forward pass is many kernels; at batch 1 the pipeline is never perfectly full, so achieved bandwidth is ~85% of peak, not 100%.

That 86%-of-ceiling result is the *good* outcome — it confirms the kernel is genuinely memory-bound and that quantization was the right lever, with the residual gap fully explained by known second-order effects. If we had measured only 2× instead of ~3.4×, the roofline would tell us to go *looking*: either the kernel is not actually memory-bound (maybe it fell back to a CPU path on an unsupported op — a classic edge gotcha) or the achieved bandwidth collapsed (uncoalesced access, KV-cache thrash). Either way, the model gives you a *prediction to falsify*, which is exactly what a scientific tool should do.

The batch-8 row makes the latency-vs-throughput point concrete: aggregate tokens/s nearly quadrupled by reusing each weight load across 8 streams (intensity rose toward the ridge), but per-stream latency did not improve — eight users each still wait roughly the single-stream time. On a single-user phone, that row is irrelevant; on a server, it is the whole game.

## When the roofline is the right tool (and when it is not)

Reach for the roofline when:

- You are about to spend real effort optimizing and want to make sure you optimize the *binding* resource. This is the default — run it first, every time.
- You are choosing between levers (quantize vs prune vs fuse) and want the data to decide rather than habit.
- You are sizing hardware: the ridge point tells you whether a candidate chip's compute-to-bandwidth ratio fits your model's intensity profile. A model that is all memory-bound decode wants a high-bandwidth chip, not a high-TOPS one.
- You want to set a *realistic* performance target: the roof is the ceiling, so it tells you the best case you could ever hit and stops you chasing impossible numbers.

Do **not** lean on it when:

- Your model is a long chain of tiny ops and launch overhead dominates — the roofline will mislead you; profile end-to-end latency and fuse the graph first.
- You are debugging *why* a kernel sits below its roof. The roofline says there is headroom; it does not say why. Occupancy, stalls, and coalescing live in the deep profiler.
- Latency, not throughput, is your single metric and your batch is fixed at 1 — the roofline's throughput framing still tells you the bandwidth floor (very useful) but cannot reason about pipeline-fill latency below that floor.
- You have not measured the real bytes. An analytical roofline built on ideal byte counts can be off by a large factor when blocking is poor; for high-stakes decisions, use hardware counters.

## Case studies and real numbers

**ShuffleNetV2 (Ma et al., 2018) — FLOPs are not latency.** The paper's central, roofline-flavored argument is that two networks with identical FLOPs can have very different latency because memory-access cost (MAC, here meaning memory accesses, not multiply-accumulates) differs. They show that the highly-fragmented, low-intensity operations that minimize FLOPs (group convolutions, excessive branching) are memory-bound and run slowly per FLOP on real GPUs and ARM CPUs. Their design guidelines — equal channel widths, fewer fragmented ops — are, in roofline terms, *raising arithmetic intensity*. This is the empirical foundation for treating low-intensity depthwise/group convs as memory-bound.

**On-device 7B LLMs — the bandwidth floor is real.** Across `llama.cpp` reports on Apple silicon and high-end Android, 4-bit 7B decode lands in the low tens of tokens per second, and the rate tracks memory bandwidth far more closely than it tracks peak compute. Going from fp16 to 4-bit yields close to the predicted ~3–4× because decode is memory-bound and quantization is a pure byte-reduction lever. This is the single most reproduced confirmation of the decode-is-memory-bound derivation.

**NVIDIA Nsight Compute's built-in roofline.** NVIDIA shipped an automated roofline analysis directly in Nsight Compute precisely because manually computing intensity per kernel is error-prone. It reads `dram__bytes` and the FLOP counters from hardware and plots each kernel against the measured roofs. The existence of this feature in a production profiler is itself evidence of how central the model is to real GPU optimization work; the Intel Advisor roofline is the CPU/Intel-GPU equivalent.

**Cache blocking turning a DRAM-bound GEMM compute-bound.** Classic HPC result, and it transfers directly to edge: a naively-written matmul re-reads operands from DRAM and sits on the slanted roof; blocking it so tiles stay in L1/L2 cuts DRAM bytes by the block factor, raising intensity until the kernel hits the flat compute roof. The hierarchical roofline is the tool that visualizes this jump between memory levels — same kernel, same FLOPs, different roof.

**FlashAttention — fusion as a roofline move.** The standard attention implementation materializes the full $S \times S$ score matrix in DRAM, reads it back to apply softmax, writes it again, and reads it once more for the value matmul — several full round-trips of an $S^2$ tensor, which at long sequence lengths is enormous bandwidth and makes attention memory-bound. FlashAttention's contribution is, in roofline language, pure denominator reduction: it tiles the computation and fuses the score-softmax-value chain so the $S \times S$ scores *never touch DRAM*, only on-chip SRAM. The FLOPs are essentially unchanged; the DRAM bytes drop by a large factor; intensity rises and the kernel climbs toward the compute roof. It is the textbook demonstration that for a memory-bound op the win comes from keeping data on-chip, exactly as the hierarchical roofline predicts — and a reminder that "is this op memory-bound?" is the question that pointed an entire research direction at the right lever.

**MCUNet and microcontroller inference — the SRAM roof.** On a Cortex-M microcontroller there is no DRAM at all in the usual sense; the working set must fit in a few hundred KB of on-chip SRAM, and the "bandwidth" that matters is to flash and SRAM. MCUNet's co-design of a tiny network and a memory-aware inference engine (TinyEngine) is, at heart, a roofline-and-memory-budget exercise: pick layer shapes whose peak activation working set fits the SRAM budget, and schedule/tile so the bandwidth-bound depthwise stages stay resident. The arithmetic-intensity reasoning is identical to the phone case; only the relevant memory level and its size change. It is the clearest proof that the roofline's "which memory level binds me?" question scales all the way down to a 256 KB device.

## Key takeaways

- **Compute one number — arithmetic intensity $I = \text{FLOPs}/\text{bytes moved}$ — and compare it to the hardware's ridge point $I^{*} = P_{\text{peak}}/B_{\text{peak}}$.** Below the ridge you are memory-bound; above it, compute-bound. That comparison is the whole diagnosis.
- **The roofline equation is $P = \min(P_{\text{peak}}, I \cdot B_{\text{peak}})$:** a flat compute roof and a slanted bandwidth roof meeting at the ridge. Your kernel sits under the lower one.
- **Edge silicon has a ridge point in the hundreds of FLOP/byte**, so most lightweight kernels — depthwise convs, activations, single-token LLM decode — are memory-bound, the opposite of what FLOP-counting suggests.
- **Single-token LLM decode is memory-bound with $I \approx 1$–$4$ FLOP/byte** because you stream every weight from DRAM per token; this sets a hard per-token latency floor of bytes-streamed divided by bandwidth, and is why 4-bit weight quantization gives a near-linear ~4× decode speedup.
- **The roofline picks the lever:** memory-bound → cut bytes (quantize, fuse, batch); compute-bound → cut FLOPs (prune, distill) or get faster MACs. Pulling the wrong lever wastes a week.
- **Quantizing fp32 → int8 cuts bytes 4× and slides a memory-bound kernel up the slanted roof for up to 4× speed**, capped when it crosses the ridge — and on a memory-bound kernel the extra dequant FLOPs are free.
- **Batching is a throughput lever, not a latency lever:** it raises intensity by reusing weight loads across streams, helping a server but not a single-user device.
- **The roofline is triage, not diagnosis:** it ignores latency, occupancy, op-launch overhead, and assumes you measured the right (DRAM) bytes. Use it to choose the lever; use the deep profiler (Nsight Compute, `torch.profiler`) to land it. Always prefer hardware byte counters over analytical estimates for high-stakes calls.

## Further reading

- Williams, Waterman, and Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, 2009 — the founding paper; read it once in full.
- Yang, Williams, et al., work on the **hierarchical roofline** and instruction-level roofline (LBNL / NERSC) — extends the model to multiple memory levels and to non-FLOP instruction mixes; the basis for the cache-vs-DRAM rooflines.
- NVIDIA Nsight Compute documentation, the "GPU Speed of Light Roofline Chart" and `--set roofline` workflow; and Intel Advisor's automated roofline guide — the production tooling that draws your kernels for you.
- Ma, Zhang, Zheng, and Sun, "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," ECCV 2018 — the empirical case that FLOPs are not latency and that memory access cost (low intensity) governs real edge speed.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame the roofline feeds, and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how this diagnostic slots into an end-to-end workflow.
- For the LLM-specific mechanics this post leans on: [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management), [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques), and [how quantization works: GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded).
