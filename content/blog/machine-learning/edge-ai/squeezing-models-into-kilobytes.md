---
title: "Squeezing models into kilobytes: MCUNet, TinyNAS, and patch-based inference"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Get ImageNet-grade vision running in 256 KB of SRAM by co-designing the architecture and the inference engine for the memory budget — the science behind MCUNet, TinyNAS, and patch-based inference, with runnable code and measured numbers."
tags:
  [
    "edge-ai",
    "model-optimization",
    "mcunet",
    "tinyml",
    "microcontrollers",
    "patch-based-inference",
    "neural-architecture-search",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/squeezing-models-into-kilobytes-1.png"
---

A microcontroller (MCU) is the cheapest, most numerous computer on earth. There are tens of billions of them in light switches, motor controllers, doorbells, hearing aids, and disposable medical patches. A typical one — a Cortex-M7 at 480 MHz, say the STM32H7 — has on the order of **512 KB of on-chip SRAM** and **1–2 MB of flash**, costs a few dollars, and sips milliwatts. For most of the deep learning era, the conventional wisdom was simple: you do not run a real convolutional network on one of these. You run a tiny keyword spotter or an accelerometer gesture classifier, and anything resembling ImageNet-grade image classification stays on the phone or in the cloud.

Then in 2020 a group at MIT showed an ImageNet classifier hitting roughly 60% top-1 accuracy running entirely inside **256 KB of SRAM and 1 MB of flash** on a commodity microcontroller. Not a demo with a GPU hiding under the table — a self-contained model and runtime on a chip that costs less than a sandwich. The system was **MCUNet**, and the way they pulled it off is one of the cleanest illustrations in all of edge ML of a single principle: **you cannot optimize the model in isolation from the engine that runs it, and on a microcontroller the thing you are optimizing is not FLOPs or parameters — it is peak memory.**

The wall here is not parameter count and it is not flash. Both of those are comfortably solvable with int8 quantization (see [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the full set of levers). The wall is **peak activation memory** — the largest set of feature-map tensors that have to be alive in SRAM at the same moment — and it is dominated by the early, high-resolution layers of the network. Figure 1 shows the shape of the problem before we do anything to it: parameters are spread across the whole network and lean toward the late layers, but peak activation memory is brutally front-loaded. That single mismatch is the entire reason MCUNet exists, and the reason its second version reaches for the cleverest trick in the post — **patch-based inference**.

![A two-panel comparison showing that network parameters concentrate in the late layers while peak activation memory concentrates in the early high-resolution stages](/imgs/blogs/squeezing-models-into-kilobytes-1.png)

By the end of this post you will be able to: reason quantitatively about why the early layers blow your SRAM budget; explain what TinyNAS searches for and why it searches the architecture **and** the input resolution **and** the channel width jointly under a memory constraint; derive the peak-memory reduction that patch-based inference buys you and the recompute cost you pay for it; sketch the patch-inference loop and the budget-constrained search in code; and decide — honestly — when full hardware-aware co-design is worth the engineering and when you should just buy a bigger chip. This post builds directly on [TinyML on microcontrollers](/blog/machine-learning/edge-ai/tinyml-on-microcontrollers) and [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint); if you have not internalized why peak activation, not parameters, is the binding limit, read that one first — everything here stands on it.

## The SRAM bottleneck, made quantitative

Let us be precise about what memory we are even talking about, because "the model is too big" hides three completely different budgets that live in different places and behave differently.

There are three storage classes on an MCU, and they map to three numbers you must track separately:

- **Flash (read-only program memory).** This holds the **weights**. After int8 quantization a model with $P$ parameters needs about $P$ bytes here, plus the code. On a 1 MB flash part you have room for roughly a million int8 weights. This is rarely the binding constraint for tiny vision models.
- **SRAM (read-write working memory).** This holds the **activations** — the intermediate feature maps — plus the stack and a little bookkeeping. This is where the model lives *while it runs*. It is small (hundreds of KB) and it is the wall.
- **Latency / compute (the MAC throughput).** A Cortex-M7 with the DSP extensions does maybe a few hundred million MACs per second of useful int8 work. This sets how many frames per second you get.

The reason peak activation is the binding constraint deserves a derivation, not an assertion. Consider a single convolution that reads an input feature map and writes an output feature map. The input tensor has shape $H_{in} \times W_{in} \times C_{in}$ and the output has shape $H_{out} \times W_{out} \times C_{out}$. While that convolution executes, **both tensors must be resident in SRAM at once** — the engine is reading from one and writing to the other. So the working set at that layer is at least

$$
M_{\text{layer}} = \underbrace{H_{in} W_{in} C_{in}}_{\text{input act}} + \underbrace{H_{out} W_{out} C_{out}}_{\text{output act}}
$$

bytes (in int8, one byte per element). The **peak activation memory** of the whole network is the maximum of this quantity — more precisely, the maximum over the execution schedule of the sum of all simultaneously-live tensors, which for a simple feedforward chain reduces to the largest adjacent input/output pair:

$$
M_{\text{peak}} = \max_{\text{layer } \ell} \big( \text{bytes of all tensors live during } \ell \big).
$$

Now watch what happens across a typical mobile-style backbone. The early layers operate on a large spatial grid — say $112 \times 112$ or $80 \times 80$ — with a modest channel count. The late layers operate on a tiny grid — $7 \times 7$ or $4 \times 4$ — with many channels. The product $H \cdot W \cdot C$ is what matters for memory, and the spatial term shrinks **quadratically** while the channel term grows only **linearly**. The early layers win the memory contest by a wide margin.

#### Worked example: where the activation peak actually sits

Take a MobileNetV2-style network fed a $160 \times 160 \times 3$ image, all activations int8 (one byte each). Walk the first few stages:

- **Input + stem (stride-2 conv to 32 channels):** input is $160 \times 160 \times 3 = 76{,}800$ bytes; output is $80 \times 80 \times 32 = 204{,}800$ bytes. Working set $\approx 281$ KB.
- **First inverted-residual block (expand 32→96 at $80\times80$):** the *expanded* intermediate tensor is $80 \times 80 \times 96 = 614{,}400$ bytes — over half a megabyte for one tensor. Working set, counting the depthwise input and output, easily exceeds **600 KB**.
- **Stage at $40\times40\times144$:** $230{,}400$ bytes for the expanded tensor; working set on the order of **300 KB**.
- **Stage at $20\times20\times192$:** $76{,}800$ bytes; working set around **120 KB**.
- **Stage at $10\times10\times384$:** $38{,}400$ bytes; under **80 KB**.
- **Final $5\times5\times1280$ + pooled head:** $32{,}000$ bytes, then a few KB after global pooling. Under **40 KB**.

The peak is the first expanded block, and it is roughly **8× larger** than the late-stage working set. Meanwhile the parameters tell the opposite story: that first block has a few thousand weights, and the classifier head alone has over a million. So you have a network whose *parameters* are 65% in the back and whose *memory* is 85%+ in the front. Figure 1 is exactly this imbalance. The headline consequence: if you only ever look at parameter count or FLOPs, you will completely misjudge whether a model fits, and you will optimize the wrong end of the network.

It helps to see the whole per-stage profile in one place, because the *shape* of the curve is the entire argument and a single peak number hides it. Here is the same network laid out stage by stage, with the working set (input act + output/expanded act, int8) next to the parameter count for that stage, so you can watch the two budgets diverge:

| Stage | Spatial × channels | Working set (int8) | Params in stage | Share of FLOPs |
| --- | --- | --- | --- | --- |
| Stem conv (stride 2) | $80\times80\times32$ | ~281 KB | ~900 | ~6% |
| IR block 1 (expand 32→96) | $80\times80\times96$ | **~384 KB (peak)** | ~5 K | ~22% |
| IR stage 2 | $40\times40\times144$ | ~300 KB | ~30 K | ~24% |
| IR stage 3 | $20\times20\times192$ | ~120 KB | ~90 K | ~20% |
| IR stage 4 | $10\times10\times384$ | ~78 KB | ~250 K | ~16% |
| Head + classifier | $5\times5\times1280$ → pool | ~38 KB | **~1.3 M** | ~12% |

Read the table top to bottom and the divergence is stark: the working-set column falls by **10×** from the first inverted-residual (IR) block to the head, while the parameter column *rises* by more than **250×** over the same span. The FLOPs are spread fairly evenly — they track the product of resolution and channels, so the early high-resolution stages and the late wide stages contribute comparably. This is the profile that every figure and every optimization in the post is reacting to. If you sized your chip by parameters you would buy SRAM for the head and starve the stem; if you sized it by FLOPs you would split the difference and still miss the peak. Only the working-set column tells you the truth, and it says: the wall is IR block 1, at ~384 KB, and nothing else is close.

There is a second-order effect in that table worth flagging because it bites people building their first MCU model. The working set for IR block 1 is *not* simply the expanded tensor's 614 KB — it is lower, ~384 KB, because a well-scheduled engine never holds the depthwise input, the full expanded tensor, and the projection output all live at once. It holds the expanded tensor and one of its neighbors. But it is also not as low as the naive "largest single tensor" guess, because the block's *input* (the stem output, $80\times80\times32 = 204$ KB) is still live as the skip path until the block's projection completes. So the realized peak is "expanded interior tensor, scheduled down, plus whatever the residual pins" — which is why a careful planner lands near 384 KB rather than either 614 KB (too pessimistic) or 153 KB (too optimistic, ignores the skip). The exact number depends on the engine's scheduler, which is precisely why you measure it on the engine you will ship, not on a spreadsheet.

This is the foundational fact the rest of the post attacks from two angles. **TinyNAS** attacks it by *designing the network* so the early-stage peak is as small as possible while keeping accuracy — which mostly means controlling input resolution and early-stage width. **Patch-based inference** attacks it by *changing how the early stage is executed* so the giant high-resolution tensor never fully materializes. And **TinyEngine** makes both pay off by generating kernels that do not waste a single byte of SRAM on interpreter overhead or unnecessary buffers. The genius of MCUNet is doing all three at once, which is why no one of them alone gets you to 256 KB.

Two more numbers to keep in your head, because they frame every decision below. First, a generic interpreter-based runtime like TensorFlow Lite for Microcontrollers (TFLite-Micro, "TFLM") carries real overhead: a tensor arena that is sized for the worst case, op records, and a memory plan that is conservative. A model whose *theoretical* peak activation is 300 KB can need well over 512 KB once TFLM's bookkeeping and conservative buffer reuse are layered on. Second, the gap between "fits on paper" and "fits in practice" is exactly the gap that a co-designed engine closes. Hold those two thoughts; they are the reason "swap in a smaller MobileNet" does not solve this and "co-design the engine too" does.

### Why int8 is the floor, not a nice-to-have

Every byte count above assumed **one byte per activation element** — that is, int8. That assumption is doing heavy lifting, and it is worth being explicit about why on an MCU int8 is not an optimization you reach for late but the baseline you start from. In fp32 the same $80\times80\times96$ tensor is not 614 KB, it is $614 \times 4 = 2.4$ MB — five times your entire SRAM. Even fp16 doubles every number above. There is no version of MCU vision that runs in fp32; the activation memory simply does not exist on the chip. So the first thing that happens, before any architecture cleverness, is that weights *and* activations go to int8. This is not the "quantize at the end to save a bit of space" story you might know from server deployment — on an MCU, int8 is the price of admission, and everything in this post is computed in that regime. (For the full story of how int8 quantization works and what it costs in accuracy, see [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq); the short version is that for these small CNNs, per-channel weight quantization plus a representative calibration set typically costs one to two points of top-1, which is acceptable when the alternative is "does not run.")

There is a subtle interaction between quantization and the memory math that trips people up. The quantization *scale and zero-point* parameters are tiny — a few bytes per tensor — so they do not move the memory needle. But the **requantization** step between layers (rescaling int32 accumulators back to int8) is an op that the engine must fuse into the convolution, or it lands an int32 intermediate in SRAM. An int32 intermediate at the early-stage resolution would be **four bytes** per element again — re-inflating the very tensor you just shrank. This is one more reason the engine matters: TinyEngine fuses requantization into the conv so the int32 accumulator lives only in registers and the accumulator-width blow-up never reaches SRAM. Get that wrong in a generic runtime and your "int8 model" briefly holds an int32 feature map at peak, and you are back over the wall.

### The peak as a graph-coloring problem

It is worth seeing the general shape of the peak-memory problem, because it explains why a good engine can beat a naive one even on the *same* architecture. Think of each activation tensor as having a **lifetime** — the interval from the step that produces it to the step of its last consumer. Two tensors whose lifetimes overlap must occupy different memory; two tensors whose lifetimes are disjoint can share the same bytes. Finding the minimum SRAM that fits all tensors is then a register-allocation / interval-graph-coloring problem, and the peak is the maximum number of bytes whose lifetimes mutually overlap at any instant.

For a straight feedforward chain this collapses to "the largest adjacent input/output pair," which is the formula we used. But real networks have residual branches: a tensor produced at the start of a block is held alive until the add at the end of the block, *across* the block's interior layers. That long-lived skip tensor sits in SRAM the whole time, stacking on top of whatever the interior layers need. So the true peak of an inverted-residual block is often "the expanded interior tensor **plus** the still-live skip input," which is exactly why the early blocks are even worse than the naive per-layer estimate suggests. A planner that understands lifetimes can sometimes reorder or recompute to shrink the overlap; a planner that does not just sums conservatively. This lifetime view is the foundation that [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) builds out in full, and it is the reason the simple analyzer later in this post is a *lower bound* on the real peak, not the exact number.

## MCUNet: TinyNAS plus TinyEngine, designed together

MCUNet (Lin et al., NeurIPS 2020, *"MCUNet: Tiny Deep Learning on IoT Devices"*) is not a model. It is a **system** with two halves that are searched and tuned jointly:

1. **TinyNAS** — a two-stage, memory-budget-aware neural architecture search that searches not just the layer structure but also the **input resolution** and the **channel width multiplier**, all under a hard SRAM and flash constraint.
2. **TinyEngine** — a code-generating inference library that emits specialized C for exactly the model you are deploying, with no interpreter, in-place depthwise convolutions, operator fusion, and memory scheduling tuned to the searched architecture.

Figure 2 shows why these belong together. TinyNAS searches in a space whose feasibility is *defined by* what TinyEngine can run cheaply; TinyEngine's measured per-layer SRAM is fed back as the constraint the search must satisfy. Run TinyNAS against a generic interpreter and the search space it can afford collapses, because the interpreter's overhead eats the budget. Run TinyEngine under a hand-designed MobileNet and you get a faster MobileNet, but not one whose memory profile was ever shaped to the chip. You need both, pointed at the same budget number.

![A dataflow graph showing a shared memory budget feeding both TinyNAS architecture search and the TinyEngine code-generating runtime, which merge into a deployable MCUNet model](/imgs/blogs/squeezing-models-into-kilobytes-2.png)

### Why search the input resolution and width, not just the layers

This is the part people miss. Classic NAS (see [neural architecture search basics](/blog/machine-learning/edge-ai/neural-architecture-search-basics)) searches the *operations* — which kernel size, which expansion ratio, how many blocks. But on an MCU the two single biggest levers on peak activation are not inside the block at all:

- **Input resolution $R$.** Peak activation in the early stage scales like $R^2$. Dropping the input from $224\times224$ to $144\times144$ cuts the early-stage feature-map area by roughly $(144/224)^2 \approx 0.41$ — a 2.4× reduction in the dominant memory term — before you touch a single layer. Resolution is the highest-leverage knob you have.
- **Width multiplier $w$.** Scaling all channel counts by $w$ scales activation memory linearly and parameter/FLOP count quadratically. A smaller $w$ in the early stages directly shrinks the peak.

A NAS that holds $R$ and $w$ fixed and only juggles block structure is searching in a space where most architectures simply do not fit, and the ones that do all look alike. TinyNAS instead treats $R$ and $w$ as **first-class search dimensions**, jointly. This is the hardware-aware NAS philosophy taken to its memory-constrained extreme — for the latency-objective version of the same idea see [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas).

### The budget-constrained search, stated formally

Before the cleverness, write down what is actually being optimized, because the constraint is the whole point. Let an architecture be described by a configuration $\theta$ that includes the input resolution $R$, the width multiplier $w$, and the per-block choices (kernel sizes, expansion ratios, depths). Let $A(\theta)$ be the model's accuracy, $S(\theta)$ its measured peak SRAM on the target engine, and $F(\theta)$ its int8 flash footprint. The search you want to solve is

$$
\max_{\theta}\; A(\theta) \quad \text{subject to}\quad S(\theta) \le S_{\max},\;\; F(\theta) \le F_{\max},
$$

where $S_{\max}$ and $F_{\max}$ are the chip's hard limits — say 256 KB and 1 MB. The thing to notice is that this is a *constrained* problem, not a penalized one. The naive approach folds the constraint into the objective as a soft penalty, $\max_\theta\, A(\theta) - \lambda\,\max(0,\,S(\theta)-S_{\max})$, and lets the search drift toward feasibility. That wastes almost the entire search budget exploring $\theta$ with $S(\theta) > S_{\max}$ — models that, no matter how accurate, **cannot be deployed at all**. On a server where "a bit over budget" means "a bit slower," soft penalties are fine. On an MCU, "over budget" means "does not boot," so the constraint must be a hard feasibility gate, evaluated *before* accuracy is ever measured. The entire architecture of TinyNAS is built to make almost every candidate it ever scores already satisfy $S(\theta)\le S_{\max}$ and $F(\theta)\le F_{\max}$, so the expensive accuracy evaluations are never spent on the infeasible region. That reframing — feasibility-first, accuracy-second — is the difference between a search that finishes and one that burns GPU-weeks on models that will never run.

### TinyNAS stage one: optimize the search space itself

Here is the genuinely clever move. Most NAS work fixes a search space and searches inside it. TinyNAS adds a stage *before* that: it **searches for the best search space** for the given budget. The intuition is that under a fixed FLOP/memory budget, the right family of (resolution, width) settings determines how good the *best possible* model in that space can be — and you can estimate that without fully training anything.

The trick they use is the **FLOPs CDF**. For a candidate search space (a fixed $R$ and $w$, with the block choices free), they randomly sample many sub-networks that satisfy the memory constraint and compute the cumulative distribution of their FLOPs. The insight: under a fixed memory budget, a search space whose feasible models tend to have **more** FLOPs is a better space, because for these tiny models FLOPs correlate strongly with accuracy and more FLOPs at the same memory means the budget is being used more productively. So TinyNAS picks the $(R, w)$ configuration whose feasible-model FLOPs CDF is shifted rightmost — the space that lets you spend the most useful compute inside the same SRAM ceiling. Figure 3 lays out this two-stage flow as a timeline.

![A five-step timeline showing TinyNAS first sizing the search space to the memory budget using a FLOPs feasibility distribution, then training and evolving sub-networks that all fit](/imgs/blogs/squeezing-models-into-kilobytes-3.png)

### TinyNAS stage two: search inside the feasible space

Once the search space is fixed to one that respects the budget, stage two is a fairly standard one-shot / weight-sharing NAS: train a single over-parameterized **super-network** in which every candidate sub-network shares weights, then run an evolutionary search over sub-networks, evaluating each by its (shared-weight) accuracy and keeping only those whose measured SRAM and flash fit. Because the space was already shaped to the budget, **every sampled subnet fits** — the search never wastes time on infeasible candidates. That is the payoff of doing stage one first.

### TinyEngine: the engine half of the co-design

A searched architecture is only as good as the kernels that run it. TinyEngine (the runtime shipped with MCUNet) is a **code generator**, not an interpreter. Instead of a general runtime that, at inference time, looks up each op, dispatches to a generic kernel, and manages a worst-case tensor arena, TinyEngine compiles the *specific* model into flat C with the schedule baked in. The wins, each of which directly attacks the SRAM wall or the latency:

- **No interpreter, no op records.** A general runtime stores metadata for every operator and a memory plan it interprets at runtime. TinyEngine emits straight-line code, so that bookkeeping memory disappears entirely from SRAM and the dispatch overhead disappears from latency.
- **In-place depthwise convolution.** A depthwise conv processes channels independently. With careful scheduling you can overwrite the input buffer as you produce the output, so the depthwise layer needs roughly **one** feature-map buffer instead of two — a large saving precisely in the high-resolution early stage where it matters most.
- **Operator fusion.** Fuse conv + batch-norm + ReLU into one pass so the intermediate never lands in SRAM; fuse the add in a residual block.
- **Specialized, loop-unrolled kernels.** Because the engine knows the exact shapes and the int8 layout at compile time, it emits kernels tuned to those shapes (im2col-free convolutions, the right tiling for the M7's cache and SIMD/DSP path), often calling into CMSIS-NN-style primitives.
- **A tight, model-specific memory plan.** Knowing the full schedule, the engine computes exactly which buffers can overlap and sizes the arena to the *true* peak, not a conservative bound.

The combined effect is dramatic: in the MCUNet paper, swapping TFLM for TinyEngine under the same model cut peak memory enough to fit models that simply would not run before, and sped inference up by roughly 3×. The architecture search then exploits that headroom — it can afford a more capable network because the engine left more SRAM on the table. That feedback loop, drawn in Figure 2, is the whole idea: **the engine's efficiency becomes search headroom, and the search's structure becomes something the engine can run cheaply.**

### What the in-place depthwise actually looks like

The in-place depthwise convolution is worth slowing down on, because it is the single trick that most directly attacks the early-stage peak, and because it shows why generated code beats an interpreter. A depthwise conv applies one small kernel per channel, and channels are independent. Naively you allocate an output buffer the same size as the input and write into it, holding both buffers — two feature maps live. But because each output element depends only on a $k \times k$ neighborhood of the *same channel*, you can process the map in a sliding window and overwrite input rows that no future output will read. With a kernel of height $k$ and stride 1, once you have produced output row $r$, you no longer need input rows above $r - k + 1$; you can reuse those rows for output. With a small ring buffer of $k$ rows you can run the entire depthwise in place, needing essentially **one** feature-map buffer plus a few rows of scratch instead of two full buffers.

In a generic interpreter this is hard to do safely — the runtime does not know, in general, that the consumer of the output never re-reads the input, so it conservatively keeps both. A code generator that has the whole graph in front of it *can* prove the input is dead after the depthwise and emit the in-place version. Here is the shape of the generated kernel, in C, to make it concrete:

```c
// In-place depthwise 3x3, stride 1, int8, with per-channel requant fused in.
// `buf` holds the input map and is overwritten with the output map in place.
// A k-row ring of scratch keeps the rows we still need before overwriting.
void dw_conv3x3_inplace_int8(
    int8_t *buf,                 // H*W*C, channel-last, overwritten
    const int8_t *weights,       // 3*3*C depthwise kernels
    const int32_t *bias,         // C
    const int32_t *out_mult,     // C  per-channel requant multiplier
    const int32_t *out_shift,    // C  per-channel requant shift
    int H, int W, int C)
{
    int8_t ring[3][/*W*C*/ 0];   // sized to 3 input rows at codegen time
    for (int y = 0; y < H; ++y) {
        save_rows_we_still_need(ring, buf, y, W, C);   // copy out before clobber
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                int32_t acc = bias[c];
                // 3x3 window from the ring (handles overwritten rows).
                acc += dot3x3(ring, weights, x, c, W, C);
                // Requant: int32 accumulator -> int8, fused, stays in registers.
                int32_t q = requantize(acc, out_mult[c], out_shift[c]);
                buf[(y * W + x) * C + c] = (int8_t)clamp_i8(q);  // write in place
            }
        }
    }
}
```

Two things to notice. First, the int32 accumulator `acc` and the requantized value `q` never leave registers — they are computed and immediately narrowed to int8, so the int32 width never costs SRAM. Second, the output overwrites `buf` while a tiny ring buffer preserves only the few input rows still needed. That is the difference between "two 600 KB buffers" and "one 600 KB buffer plus a few KB of ring" at the early stage. On a 256 KB chip, that difference is the difference between running and not running. The actual TinyEngine kernels go further — calling into CMSIS-NN SIMD primitives for the inner dot product on Cortex-M parts with the DSP extension — but the memory structure is exactly this.

### Why fusion matters more on an MCU than on a GPU

On a GPU, operator fusion is mostly a *bandwidth* win: you avoid a round trip to HBM. On an MCU it is a *capacity* win: the fused-away intermediate would have had to fit in SRAM, and there often is not room. Fusing conv + batch-norm + ReLU + requantize into one pass means the only tensors that ever touch SRAM are the conv's input and its final int8 output — none of the three intermediate forms (post-conv int32, post-BN, post-ReLU) ever materialize. On a chip where the difference between fitting and not is tens of KB, removing an intermediate feature map from the schedule is not a speed tweak, it is a feasibility requirement. This is the same graph-level reasoning covered in [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization), pushed to the regime where the stakes are "does it run at all."

## MCUNetV2 and the key trick: patch-based inference

MCUNet V1 got an ImageNet classifier into 256 KB by *designing around* the early-stage memory peak — small resolution, narrow early stages. But that design choice costs accuracy: a low input resolution throws away detail. The follow-up, **MCUNetV2** (Lin et al., NeurIPS 2021, *"MCUNetV2: Memory-Efficient Patch-Based Inference for Tiny Deep Learning"*), asks a sharper question: *what if we did not have to materialize the giant early-stage feature map at all?* If the early-stage peak were not the wall, we could feed a higher-resolution image and recover the accuracy.

That is exactly what patch-based inference does. Figure 4 contrasts the two execution modes.

![A two-panel comparison of whole-map inference holding one large feature map live versus patch-based inference processing small tiles one at a time](/imgs/blogs/squeezing-models-into-kilobytes-4.png)

### The idea

A convolution is **local**: each output pixel depends only on a small neighborhood of the input (its receptive field). That means you do not have to compute the whole output feature map before moving on — you can compute it **region by region**. Patch-based inference exploits this in the memory-heavy early stage:

1. Split the output of the early stage into a grid of small spatial **patches** (tiles) — say a $2\times2$ or $4\times4$ grid.
2. For each output patch, compute *only* the input region it depends on, run the early-stage layers on just that region, and produce just that output patch.
3. Hand each completed patch to the rest of the network (the later, low-resolution stages run normally, on the whole small map).

Because you process **one patch at a time**, the largest tensor that is ever live in SRAM is one patch's worth of activation, not the whole feature map. The full high-resolution feature map is never simultaneously resident — it exists only as a sequence of small tiles passing through.

### Deriving the peak-memory reduction

Let the early stage produce a feature map of spatial size $H \times W$ with $C$ channels, and split it into an $n \times n$ grid of patches. Ignore the overlap for a moment. The whole-map peak for that stage is

$$
M_{\text{whole}} \approx H \cdot W \cdot C \quad (\text{plus the input map of similar size}).
$$

With $n \times n$ patches, each patch's output is about $\frac{H}{n} \times \frac{W}{n} \times C$, and only one patch is live at a time, so the per-patch peak is

$$
M_{\text{patch}} \approx \frac{H}{n} \cdot \frac{W}{n} \cdot C = \frac{H \cdot W \cdot C}{n^2}.
$$

The reduction factor is therefore approximately $n^2$ — a $2\times2$ grid cuts the early-stage peak by ~4×, a $3\times3$ grid by ~9×. In practice the overlap and the per-tile inputs blunt this, so the *realized* peak reduction reported in MCUNetV2 is around **4–8×** for the early stage, which is exactly the order of magnitude needed to drop a 384 KB peak under a 256 KB ceiling. Figure 7 shows the before/after on the peak directly. Critically, this lets you *raise the input resolution* — the thing V1 had to sacrifice — and recover several points of accuracy at the same SRAM budget.

### The catch: receptive-field overlap and recompute

There is no free lunch, and the price here is **recompute**. Each output patch depends on a slightly *larger* input region than its own footprint, because the convolutions in the early stage have a receptive field bigger than one pixel. Adjacent patches therefore share a border — a **halo** — of input that has to be processed for *both* of them. The deeper the early stage and the larger the kernels, the wider the halo, and the more the patches overlap. Figure 5 shows the halo structure for a $2\times2$ grid: the center region is shared by all four tiles.

![A three-by-three grid showing four corner output tiles sharing halo regions along their edges and a center region overlapped by all four tiles](/imgs/blogs/squeezing-models-into-kilobytes-5.png)

If the early stage has total stride $s$ and the patches are small, the halo can be a significant fraction of each patch, and naively the recompute overhead can be large — in the worst case the per-patch input regions sum to far more than the original input. MCUNetV2's second contribution is the fix: **receptive-field redistribution.** They restructure the network so the early (patched) stage has a **smaller** receptive field — fewer / smaller early downsampling layers, less aggressive early striding — which shrinks the halo and therefore the recompute. The receptive field that was removed from the early stage is **added back later**, in the low-resolution stages that run whole-map and where extra receptive field is essentially free (the feature map is already tiny). With this redistribution, the recompute overhead drops to roughly **10–20%** of FLOPs — a cost you happily pay to get a 4–8× peak-memory cut and the higher resolution it unlocks.

That trade is the heart of the trick: you spend a little extra compute to buy a lot of peak-memory headroom, and you arrange the architecture so the extra compute stays small. On an MCU, where you are memory-bound long before you are compute-bound, that is exactly the right trade. (For the general principle of which resource binds you — memory bandwidth versus arithmetic — see [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives).)

### Quantifying the halo

Let us put numbers on the overlap so the recompute cost is not hand-waved. Suppose the patched early stage has total receptive field $r$ — that is, each output pixel depends on an $r \times r$ window of the stage input. If you split the stage output into tiles of side $t$ (in output pixels), each tile needs an input window of side $t \cdot s + (r - s)$ where $s$ is the total stride, because the tile must see the receptive field of its border pixels. The halo width on each side is roughly $(r - s)/2$ in input pixels. The fraction of recomputed input is then approximately

$$
\text{overhead} \approx \left(\frac{t \cdot s + (r - s)}{t \cdot s}\right)^2 - 1 \approx \frac{2(r-s)}{t \cdot s}
$$

for small overhead. The lesson is in the algebra: the overhead grows with the receptive field $r$ and *shrinks* with the tile size $t$. So there are two levers to keep recompute small — make the early-stage receptive field small (receptive-field redistribution), and do not make the tiles tiny. If $r$ is small (say the early stage only downsamples once or twice), the halo is a thin border and the overhead stays in the low tens of percent even for a $2\times2$ grid. If you let $r$ grow large by stacking aggressive early downsampling, the halo can swallow the tile and the overhead explodes — which is exactly the failure MCUNetV2's redistribution exists to prevent.

To make the algebra land, put it in a small table. Fix the total stride at $s = 4$ (two stride-2 downsamples in the patched stage) and vary the output tile side $t$ and the receptive field $r$. The recomputed-input fraction is $\big((t s + (r-s))/(t s)\big)^2 - 1$:

| Tile side $t$ (out px) | Receptive field $r$ | Input window side | Recompute overhead |
| --- | --- | --- | --- |
| 20 | 7 (redistributed, small) | $20\cdot4 + 3 = 83$ | ~7.7% |
| 20 | 31 (aggressive early stride) | $80 + 27 = 107$ | ~43% |
| 10 | 7 | $43$ | ~16% |
| 10 | 31 | $67$ | ~80% |
| 5 | 31 | $47$ | ~120% |

The bottom rows are the disaster the redistribution exists to avoid: with a large early receptive field and small tiles, you recompute *more than the original input*, and patch inference becomes a net compute loss while the memory win shrinks (because each tile's input window is now huge). The top row is the regime MCUNetV2 engineers the network into — small $r$, modest tile count — where the overhead is a thin border you barely notice. The same table read as a design rule: **keep $r$ small and $t$ from getting tiny**, which means redistribute receptive field out of the patched stage and never split into more tiles than the budget actually requires. The recompute overhead is not a property of "patching" in the abstract; it is a property of the receptive field of the layers you chose to patch, and that is a knob you control.

#### Worked example: when the halo eats the tile

Suppose an engineer, not knowing the redistribution trick, naively applies patch inference to the *first three* stages of a MobileNetV2 — three stride-2 downsamples, $s = 8$, with $3\times3$ kernels stacking to a receptive field around $r \approx 35$ input pixels. They pick an aggressive $4\times4$ grid to crush the peak, so the output tile side is small, $t \approx 5$. Plug in: the input window per tile is $t s + (r - s) = 40 + 27 = 67$ pixels on a side, against an "own footprint" of $t s = 40$. The recompute overhead is $(67/40)^2 - 1 \approx 1.8$ — they are doing **almost three times** the early-stage compute. Worse, the per-tile input window is $67\times67$, not the $40\times40$ they budgeted for, so the per-tile *memory* is more than double what the clean $n^2$ formula promised. They went looking for an 16× memory cut and a small compute tax and got a ~7× memory cut and a 180% compute blow-up — the trade collapsed. The fix is not "use fewer tiles" alone; it is restructure the network so only the first one or two layers are patched (small $r$), then the same $4\times4$ grid behaves like the top row of the table. This is exactly the failure mode the [stress test](#stress-test) below pushes on, and the reason patches and redistribution ship as a pair.

This is why receptive-field redistribution and patch-based inference are a *package*, not two independent tricks. Patches alone, on a network with a large early receptive field, would pay ruinous recompute. Redistribution alone, without patches, just shuffles where the receptive field lives without saving memory. Together: small early receptive field makes the halo thin, thin halo makes patches cheap, cheap patches cut the peak, the cut peak buys resolution, and the late stages absorb the receptive field that was moved out of the early ones. Every piece is in service of the same memory budget. Figure 5 is the spatial picture of that thin halo for a $2\times2$ grid; the center cell is the only region all four tiles share, and keeping it small is the whole game.

## The hardware-aware co-design principle

Step back from the specifics and the lesson generalizes. The thing that made MCUNet work was refusing to treat any one layer of the stack as fixed:

- The **architecture** is searched for the memory budget (TinyNAS), including resolution and width.
- The **engine** is generated for the architecture (TinyEngine), eliminating overhead and minimizing live buffers.
- The **execution schedule** is changed (patch-based inference) so the worst tensor never fully materializes.
- The **numerics** are int8 throughout, which is what makes the byte-counting above hold and what halves the activation bytes versus fp16.

Each of these alone moves the needle a little. Together they cross the threshold from "impossible on this chip" to "ships." This is the same co-design idea that runs through the whole series — the four levers (quantization, pruning, distillation, efficient architecture) sit on top of compilers and runtimes, and you read the result off the accuracy–efficiency Pareto frontier (see [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression)). MCUNet is the most extreme point on that frontier: when the budget is 256 KB, you cannot afford to leave *any* layer of the stack un-optimized.

The counterpoint, which I will make honestly later, is that co-design is *expensive engineering*. You do it when the budget forces you to. When you have tens of MB of SRAM, the smart move is usually to apply standard quantization and a good off-the-shelf runtime and spend your time elsewhere. The decision tree in Figure 8 captures exactly that boundary.

## Practical: the tools, the patch loop, and measuring SRAM

Let us get concrete. The MCUNet and TinyEngine code is open source from the MIT Han Lab; the relevant repos are `mit-han-lab/mcunet` (the models and TinyNAS) and `mit-han-lab/tinyengine` (the code-generating runtime). The deployment path is: search or download a pretrained MCUNet model, quantize to int8, run it through the TinyEngine code generator to emit C, and compile that C for your target with the vendor toolchain.

### Pulling a pretrained MCUNet model

The `mcunet` package exposes the searched models by name and target budget, so you can pull one and inspect it without running the search yourself.

```python
# pip install mcunet  (from mit-han-lab/mcunet)
from mcunet.model_zoo import net_id_list, build_model
import torch

# List the available pretrained nets; ids encode the target budget.
print(net_id_list)  # e.g. 'mcunet-in1', 'mcunet-in2', ... 'mcunet-in4'

# Build an MCUNet targeted at a 256 KB SRAM / 1 MB flash MCU, ImageNet-pretrained.
model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)
model.eval()
print(description)            # input resolution, width, reported SRAM/Flash
print("input size:", image_size)

# A forward pass at the model's native resolution.
x = torch.randn(1, 3, image_size, image_size)
with torch.no_grad():
    logits = model(x)
print(logits.shape)           # (1, 1000)
```

The thing to notice: the model *comes with* its input resolution baked in, because resolution was a search dimension. You do not get to feed it whatever size you like — the architecture and the resolution are a matched pair chosen together for the budget.

### Measuring peak activation memory yourself

Before you trust any reported number, measure the peak activation for your own model. The honest way is to simulate the execution schedule: walk the layers, track which tensors are live, and record the maximum simultaneous bytes. Here is a simplified analyzer for a feedforward chain that captures the core idea — the live set at each op is its inputs plus its output.

```python
import torch
import torch.nn as nn

def peak_activation_bytes(model, input_shape, bytes_per_elem=1):
    """Estimate peak activation memory for a feedforward model.

    Walks modules in execution order. For a simple chain the live set at
    each op is (input tensor) + (output tensor); the peak is the max.
    bytes_per_elem=1 models int8 activations, 4 models fp32.
    """
    sizes = {}                 # module -> output element count
    peak = 0

    def hook(mod, inp, out):
        nonlocal peak
        in_elems = sum(t.numel() for t in inp if torch.is_tensor(t))
        out_elems = out.numel() if torch.is_tensor(out) else 0
        live = (in_elems + out_elems) * bytes_per_elem
        peak = max(peak, live)

    handles = []
    for m in model.modules():
        # Only leaf modules actually allocate buffers.
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape))

    for h in handles:
        h.remove()
    return peak

mcu_model, res, _ = ("...", 144, None)   # from build_model above
# peak = peak_activation_bytes(model, (1, 3, res, res), bytes_per_elem=1)
# print(f"peak activation: {peak/1024:.1f} KB (int8)")
```

This is deliberately simple — it does not model residual branches that keep a tensor live across several layers, or in-place ops that *reduce* the peak. A real memory planner (and TinyEngine's) does both: it computes each tensor's lifetime as the interval from production to last use, then finds the maximum overlap of those intervals. That is the lifetime analysis covered in depth in [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint). For a first cut, the input-plus-output peak above is within a small factor and tells you which layer is the wall — which is the number you actually act on.

### The patch-inference loop, sketched

Here is the structure of patch-based inference for the early stage. The key pieces are computing, for each output tile, the input region it needs (including the halo), running the early stage on just that region, and assembling the output tiles into the full small feature map that the rest of the network consumes whole.

```python
import torch
import torch.nn.functional as F

def patch_based_early_stage(x, early_stage, n_tiles, halo):
    """Run `early_stage` on `x` one spatial tile at a time.

    x          : (1, C, H, W) input to the memory-heavy early stage
    early_stage: nn.Sequential of the early (patched) layers; total stride S
    n_tiles    : tiles per side (n_tiles x n_tiles grid)
    halo       : extra input pixels each tile pulls from neighbors
                 (set from the early stage's receptive field)

    Returns the assembled early-stage OUTPUT. Only one tile's worth of
    activation is ever live, so the peak is ~ whole_peak / n_tiles^2.
    """
    _, _, H, W = x.shape
    S = total_stride(early_stage)            # output is H/S x W/S
    out_h, out_w = H // S, W // S
    th, tw = out_h // n_tiles, out_w // n_tiles   # output tile size

    out = None                               # lazily sized after first tile
    for i in range(n_tiles):
        for j in range(n_tiles):
            # Output tile [i,j] occupies these output coords ...
            oy0, ox0 = i * th, j * tw
            oy1, ox1 = oy0 + th, ox0 + tw
            # ... which map back to an INPUT window plus a halo.
            iy0 = max(0, oy0 * S - halo)
            ix0 = max(0, ox0 * S - halo)
            iy1 = min(H, oy1 * S + halo)
            ix1 = min(W, ox1 * S + halo)

            tile_in = x[:, :, iy0:iy1, ix0:ix1]          # one small region
            tile_out_full = early_stage(tile_in)          # one tile live here
            # Crop away the halo so tiles butt together cleanly.
            cy0 = (oy0 * S - iy0) // S
            cx0 = (ox0 * S - ix0) // S
            tile_out = tile_out_full[:, :, cy0:cy0 + th, cx0:cx0 + tw]

            if out is None:
                C_out = tile_out.shape[1]
                out = x.new_zeros(1, C_out, out_h, out_w)
            out[:, :, oy0:oy1, ox0:ox1] = tile_out
    return out

def total_stride(stage):
    s = 1
    for m in stage.modules():
        s *= getattr(m, "stride", (1,))[0] if hasattr(m, "stride") else 1
    return s
```

On a microcontroller this loop is generated as C by TinyEngine, with the halo and tile sizes resolved at compile time and the tile buffer reused across iterations — so in practice there is exactly one tile buffer in SRAM, written and overwritten `n_tiles**2` times. The Python above is the readable model of what that generated loop does. Note the halo crop: each tile is computed slightly oversized and the border trimmed, so the assembled output is seamless despite the per-tile recompute.

### Measuring the patch loop's peak directly

The whole reason to write the loop is the peak it achieves, so instrument it. The honest way on-device is the high-water-mark trick: paint the arena with a known sentinel byte before inference, run it, then scan from the top of the arena for the first byte the run touched — the distance from the arena base to that byte is the realized peak. Here is the same idea in Python, wrapping the patch loop so you can compare whole-map versus patched peak on your own model before you ever cross-compile. It tracks the largest single tensor produced inside the early stage, which on a one-tile-live schedule is the per-tile peak:

```python
import torch

class PeakTracker:
    """Record the largest activation (in int8 bytes) seen during a forward pass.
    On a one-tile-at-a-time schedule this equals the per-tile peak SRAM,
    which is the number that has to fit under the ceiling."""
    def __init__(self, bytes_per_elem=1):
        self.peak = 0
        self.bpe = bytes_per_elem
        self._handles = []

    def _hook(self, mod, inp, out):
        live = 0
        for t in inp:
            if torch.is_tensor(t):
                live += t.numel()
        if torch.is_tensor(out):
            live += out.numel()
        self.peak = max(self.peak, live * self.bpe)

    def attach(self, module):
        for m in module.modules():
            if len(list(m.children())) == 0:           # leaf modules only
                self._handles.append(m.register_forward_hook(self._hook))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

def compare_peaks(x, early_stage, n_tiles, halo):
    """Return (whole_map_peak_kb, patched_peak_kb) for one early stage."""
    # Whole-map: run the stage on the full input, record the peak.
    t_whole = PeakTracker().attach(early_stage)
    with torch.no_grad():
        early_stage(x)
    t_whole.detach()

    # Patched: the per-tile peak is the max single-tile working set. Run the
    # stage on ONE representative tile-sized input window (with halo).
    _, C, H, W = x.shape
    S = total_stride(early_stage)
    tile_out = (H // S) // n_tiles
    win = tile_out * S + 2 * halo                       # input window side
    tile_in = x[:, :, :win, :win]
    t_patch = PeakTracker().attach(early_stage)
    with torch.no_grad():
        early_stage(tile_in)
    t_patch.detach()

    return t_whole.peak / 1024, t_patch.peak / 1024

# whole_kb, patch_kb = compare_peaks(x, early_stage, n_tiles=2, halo=4)
# print(f"whole-map peak {whole_kb:.0f} KB  ->  patched peak {patch_kb:.0f} KB")
# print(f"reduction: {whole_kb / patch_kb:.1f}x")
```

Two cautions on this measurement, both of which separate a believable number from a misleading one. First, the per-tile run includes the halo (`2 * halo` added to the window), so the reported patched peak is the *realized* peak with overlap, not the optimistic $n^2$ ideal — that is what you want, because the halo is real SRAM. Second, this tracker counts input-plus-output per leaf op, which, like the analyzer earlier, ignores residual tensors pinned across a block; if your patched stage has a skip connection, add the pinned skip tensor's bytes to the reported peak by hand or the number will be too low. The point of measuring before cross-compiling is to catch exactly the "the halo ate my budget" surprise from the worked example above on your laptop, in seconds, instead of on the board after a day of toolchain wrangling.

### TinyNAS's budget constraint, in pseudocode

The core of stage-one search — pick the search space that uses the budget best — is a short loop in spirit. For each candidate (resolution, width), sample feasible subnets and rank spaces by where their FLOPs distribution sits.

```python
import numpy as np

def pick_search_space(candidate_spaces, sram_budget_kb, flash_budget_kb,
                      n_samples=1000, percentile=20):
    """Stage 1 of TinyNAS: choose the (resolution, width) search space
    whose feasible models pack the most FLOPs into the budget.

    candidate_spaces: list of dicts with 'res', 'width', and a sampler.
    We score each space by the FLOPs at a low percentile of its feasible
    distribution (a right-shifted CDF == a better space for tiny models).
    """
    best, best_score = None, -1.0
    for space in candidate_spaces:
        feasible_flops = []
        for _ in range(n_samples):
            net = space["sample"]()                 # a random subnet
            if (net.peak_sram_kb() <= sram_budget_kb and
                    net.flash_kb() <= flash_budget_kb):
                feasible_flops.append(net.flops())
        if not feasible_flops:
            continue                                # space too rich for budget
        # Right-shifted FLOPs CDF -> use more compute at same memory.
        score = np.percentile(feasible_flops, percentile)
        if score > best_score:
            best, best_score = space, score
    return best   # then run weight-sharing + evolutionary search inside it
```

The point is not the exact statistic — it is that **the memory budget is a hard filter applied before accuracy is ever considered**, and the search space itself is chosen so that almost everything inside it already passes that filter. Contrast that with bolting a memory penalty onto the loss of an otherwise-unconstrained search; that wastes most of the search budget exploring models that will never fit.

## Worked examples with real numbers

### Worked example: flattening the early-stage peak with patches

Take the MobileNetV2-style profile from earlier, fed a $160\times160$ image, int8 activations. The early-stage peak was the first expanded inverted-residual block: the expanded tensor at $80\times80\times96$ is $614{,}400$ bytes, and the whole-map working set there is on the order of **384 KB** once you count the depthwise input alongside it (the engine does not need to hold the full expand and the full depthwise output and the projection output all at once with good scheduling, so the realized peak is below the raw sum). Suppose your chip is a 256 KB-SRAM part. You are 128 KB over the wall.

Now apply patch-based inference to that early stage with a $2\times2$ grid:

- Each output tile is about $40\times40$ in the output feature space, so each tile's expanded activation is roughly $40\times40\times96 \approx 153{,}600$ bytes before overlap.
- With only one tile live and good buffer reuse, the per-tile peak lands around **56 KB** — about a **7× reduction** from the 384 KB whole-map peak.
- The halo: with a modest early receptive field after redistribution, each tile pulls a few extra rows/columns of input, adding on the order of **10–15% recompute** to the early stage's FLOPs. The early stage is a small fraction of total FLOPs, so the whole-network compute overhead is in the low single-digit percent.

Result: you drop from 384 KB (over the wall by 128 KB) to ~56 KB (200 KB of headroom) at a cost of a few percent more compute — and, crucially, that headroom is what lets you raise the input resolution back up to recover accuracy. Figure 7 is precisely this before/after on the peak. This is the single highest-leverage memory optimization in the whole MCU vision playbook.

### Worked example: MCUNet ImageNet on a 256 KB and a 512 KB MCU

These are the headline numbers from the MCUNet papers, marked approximate because exact figures depend on the specific net id, the target part, and the measurement setup — always re-measure on your own hardware before quoting.

On a **256 KB-SRAM / 1 MB-flash class MCU** (e.g. STM32F746-class), MCUNet V1 reaches roughly **60–62% ImageNet top-1** with peak SRAM around **238 KB** and an int8 model around **0.9 MB** in flash, running at a few frames per second. The directly comparable baseline — a scaled-down MobileNetV2 on TFLite-Micro — lands around **49% top-1** *if it fits at all*; at the resolution and width needed to match MCUNet's accuracy it blows the SRAM budget on that runtime, so it simply does not run. That is the gap: same chip, ~12 accuracy points, and the baseline often cannot even be deployed.

Step up to a **512 KB-SRAM** part and the budget loosens. MCUNet can run a higher-resolution / wider configuration, and **MCUNetV2** with patch-based inference pushes ImageNet top-1 to roughly **64–68%** while keeping peak SRAM well under the ceiling — because patches mean the resolution can rise without the early-stage peak rising with it. The pattern holds at both budgets: the co-designed stack converts saved SRAM into accuracy, and the patch trick converts the same SRAM into *resolution*, which is itself accuracy.

A second class of result worth knowing, from the same line of work: on **visual wake words** (the binary "is there a person?" benchmark designed for tiny vision), MCUNet/MCUNetV2 reach well over **90% accuracy** inside a few hundred KB of SRAM — a task that is genuinely useful for always-on, battery-powered sensors, and one where the memory savings translate directly into a smaller, cheaper, longer-lived device.

For how to *measure* any of these honestly on your own board — warm-up runs, batch-1 reality, peak-SRAM instrumentation rather than theoretical estimates — see [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device). The short version: report peak SRAM from the linker map and a runtime high-water mark, not from a spreadsheet; report latency after warm-up and across many runs; and never quote an accuracy number you did not re-evaluate on the quantized, deployed model.

#### Worked example: battery life of an always-on vision sensor

The reason any of this matters commercially is power, so let us close the loop with an energy budget. Take a coin-cell-powered occupancy sensor that runs an MCUNetV2 visual-wake-word model on a Cortex-M7 at 480 MHz, waking once per second to check "is a person here?" Assume the inference takes 150 ms and the M7 draws roughly 100 mW while computing, and the chip sleeps at about 50 microwatts between inferences.

- **Active energy per inference:** $100\ \text{mW} \times 0.15\ \text{s} = 15\ \text{mJ}$, which is about $4.2\ \mu\text{Wh}$ per check.
- **Sleep energy per second:** $50\ \mu\text{W} \times 0.85\ \text{s} \approx 0.012\ \mu\text{Wh}$ — essentially nothing next to the active burst.
- **Per second total:** dominated by the active burst, $\approx 4.2\ \mu\text{Wh}$, so roughly $15\ \text{mWh}$ per hour, or about $0.36\ \text{Wh}$ per day at one check per second.
- **A CR2032 coin cell** holds roughly $0.6\ \text{Wh}$ usable. At this duty cycle that is on the order of a couple of days — which is why the *real* design lowers the check rate to once every few seconds and gates the expensive vision model behind a cheap motion trigger, pushing battery life into months.

Now run the counterfactual: this device is only possible because the model fits in the MCU's SRAM at all. If it needed a Linux-class SoC to hold its activations, the idle power alone would be tens of milliwatts — three orders of magnitude more than the MCU's sleep draw — and the coin cell would die in hours, not months. The memory savings from patch-based inference are not an abstract leaderboard number; they are what lets the whole thing run on a chip that can sleep at microwatts. That is the entire commercial argument for squeezing models into kilobytes: it changes which *class* of device, and therefore which *class* of battery and product, is feasible. A precise per-inference figure here should always be measured on the real board with a power monitor, not estimated — these numbers are illustrative and the active draw varies several-fold across MCUs and clock settings.

## Results: MCUNet versus the baselines

Figure 6 puts the comparison in one matrix. The numbers below it are the same data in table form, with the explicit caveat that they are approximate and target-dependent.

![A four-by-four comparison matrix of peak SRAM, flash, ImageNet accuracy, and latency across MobileNetV2 baselines and MCUNet variants](/imgs/blogs/squeezing-models-into-kilobytes-6.png)

| System | Peak SRAM | Flash (int8) | ImageNet top-1 | Latency / runs |
| --- | --- | --- | --- | --- |
| MobileNetV2 (scaled) + TFLite-Micro | > 512 KB | ~1.0 MB | ~49% | does not fit 256 KB |
| MobileNetV2 (scaled) + TinyEngine | ~320 KB | ~1.0 MB | ~49% | ~3× faster than TFLM |
| **MCUNet V1** (256 KB target) | **~238 KB** | ~0.9 MB | **~61%** | 5–10 fps on M7 |
| **MCUNetV2** (patch-based) | **~140 KB** | ~1.0 MB | **~64%** | 5–10 fps on M7 |

Two things to read off this table. First, look at the top two rows: keeping the *model* fixed and only swapping the *engine* (TFLM → TinyEngine) takes a model from "does not fit" to "fits and runs 3× faster." That is the engine half of the co-design paying for itself before the architecture search even starts. Second, look down the SRAM column: MCUNetV2's patch-based inference cuts peak SRAM well below V1's *while raising accuracy*, because the patches buy resolution. The whole arc of the table is the co-design thesis: every column improves because every layer of the stack was optimized for the same budget.

### MCUNet versus MobileNetV2 + TFLM on a named MCU

Generic tables invite hand-waving, so pin it to a real part: the **STM32H743**, a Cortex-M7 at 480 MHz with **512 KB of on-chip SRAM** (plus tightly-coupled memory) and 2 MB of flash — a common upper-mid tinyML target. Run the same ImageNet-classification job two ways: a width-and-resolution-scaled MobileNetV2 on TFLite-Micro, versus an MCUNet searched and run on TinyEngine. These figures are approximate and follow the trends reported in the MCUNet papers; re-measure on your own board and net id before quoting.

| Config (STM32H743, M7 @ 480 MHz) | Input res | Peak SRAM | Flash (int8) | ImageNet top-1 | Latency / frame |
| --- | --- | --- | --- | --- | --- |
| MobileNetV2 ×0.35 + TFLM | $128\times128$ | ~490 KB | ~0.7 MB | ~49% | ~520 ms |
| MobileNetV2 ×0.5 + TFLM | $160\times160$ | > 512 KB | ~1.0 MB | ~54% | does not fit |
| MobileNetV2 ×0.5 + TinyEngine | $160\times160$ | ~330 KB | ~1.0 MB | ~54% | ~180 ms |
| **MCUNet V1 + TinyEngine** | $160\times160$ | **~240 KB** | ~0.9 MB | **~61%** | ~140 ms |
| **MCUNetV2 (patch) + TinyEngine** | $176\times176$ | **~160 KB** | ~1.0 MB | **~64%** | ~155 ms |

Walk it row by row, because each comparison isolates one variable. Rows 1 and 2 are MobileNetV2 on the *generic* runtime: the ×0.35 model barely fits at low resolution and stalls at 49%, and the moment you ask for the resolution and width needed to reach the mid-50s, TFLM's conservative arena pushes peak SRAM past the 512 KB ceiling and it **does not fit at all**. Row 3 keeps that exact ×0.5 model but swaps in TinyEngine: peak SRAM drops from "over 512 KB" to ~330 KB — the same architecture, fitting comfortably, running ~3× faster — purely from in-place depthwise, fusion, and a tight memory plan. Rows 4 and 5 are where the architecture search earns its keep: MCUNet V1 uses the headroom TinyEngine opened to run a *better-shaped* network at the same 160-pixel resolution, landing ~7 points above the hand-scaled MobileNet at lower peak SRAM and lower latency; MCUNetV2 then spends its patch-bought headroom on *higher resolution* (176 px), pushing accuracy to ~64% while dropping peak SRAM to ~160 KB. The latency ticks up slightly from V1 to V2 — that is the patch recompute tax, the ~10–20% overhead made visible — and it is a trade you take gladly for 3 points of accuracy and 80 KB of reclaimed SRAM. Every row changes exactly one thing, and every change moves in the direction the co-design thesis predicts.

Now the patch-versus-whole peak directly, the single most important before/after in the post:

| Execution mode (early stage) | Tensor live | Peak SRAM (early stage) | Fits 256 KB? |
| --- | --- | --- | --- |
| Whole-map | full $80\times80\times96$ map | ~384 KB | No, over by 128 KB |
| Patch-based, $2\times2$ | one $40\times40\times96$ tile | ~56 KB | Yes, ~200 KB headroom |
| Patch-based, $3\times3$ | one $\sim27\times27\times96$ tile | ~28 KB | Yes, even more headroom |

![A two-panel comparison showing whole-map early-stage peak above the SRAM ceiling and patch-based per-tile peak comfortably under it](/imgs/blogs/squeezing-models-into-kilobytes-7.png)

The recompute cost climbs with the tile count (more tiles, more shared halo), so you do not crank $n$ to infinity — you pick the smallest $n$ that gets the early-stage peak under the budget with margin, which for the 256 KB target is usually a $2\times2$ grid. Beyond that you are paying recompute for headroom you do not need.

### How to measure this honestly

A reminder, because tiny-device numbers are easy to fudge. Peak SRAM should come from two sources that must agree: the **linker map** (the static arena size TinyEngine reserved) and a **runtime high-water mark** (paint the stack and arena with a sentinel byte, run inference, scan for the lowest untouched address). If those disagree, trust the runtime mark — the static size can be conservative. Latency should be measured **after warm-up**, on a thermally settled board, at **batch 1** (the only batch size that exists on an MCU), averaged over many frames, and reported with a p99 if you care about real-time deadlines. Accuracy must be re-evaluated on the **quantized, deployed** graph, not the fp32 model — int8 PTQ on these tiny nets can cost a point or two, and you want that point counted against the deployed artifact, not hidden.

## When to reach for full co-design (and when not to)

This is the section I would want a junior engineer to read twice, because the failure mode of this whole topic is over-engineering. MCUNet is a magnificent piece of work, and it is the **wrong answer** for most edge deployments. Here is the decision boundary, which Figure 8 draws as a tree.

![A decision tree branching on SRAM budget, recommending full co-design and patch inference under sub-megabyte budgets and standard quantization or a bigger chip otherwise](/imgs/blogs/squeezing-models-into-kilobytes-8.png)

**Reach for full co-design (TinyNAS + TinyEngine + patch inference + int8) when:**

- Your SRAM budget is **sub-megabyte** — you are on a Cortex-M class MCU, not a Linux-class SoC. This is the regime where peak activation is genuinely the wall and no amount of standard quantization gets you under it.
- The device **cannot** be upgraded for cost, power, or form-factor reasons — a coin-cell sensor, a disposable medical patch, a part you are shipping by the million where a fifty-cent BOM increase is fatal.
- The task genuinely needs a real CNN (image classification, visual wake words, small object detection), not a tiny MLP on engineered features.
- You will amortize the engineering over a large fleet. Co-design is real effort; it pays off across a million units, not across a prototype.

**Do not reach for it — do something cheaper — when:**

- You have **tens of MB of SRAM or more** (a Raspberry Pi, a Jetson, a phone). Here peak activation is rarely the binding constraint; standard int8 PTQ plus a good runtime (LiteRT, ONNX Runtime, TensorRT) gets you there with a fraction of the effort. See [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) for the off-the-shelf options.
- **Stepping up the chip is cheaper than the engineering.** This is the honest one. If moving from a 256 KB MCU to a 1 MB MCU costs a dollar a unit and saves you three months of NAS-and-kernel work, *buy the bigger chip* unless your volume makes the dollar dominate. Co-design is a tool for when the hardware is truly fixed, not a badge of honor.
- A simpler model already fits. If a hand-designed MobileNetV3-Small at low resolution hits your accuracy target inside the budget on a stock runtime, ship it. Do not run a 200-GPU-hour search to shave 30 KB you did not need to shave.
- Your bottleneck is **latency or energy, not memory.** Patch inference *adds* compute; if you are already compute-bound, it can make latency worse. Co-design's memory wins only matter when memory is the wall — diagnose the bottleneck first (this is what [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) is for).

### Stress-testing the approach

Push on the edges, because that is where you learn whether you understand it:

- **What if the calibration set for int8 is tiny?** PTQ on a few hundred images is usually fine for these small CNNs, but watch the early high-resolution layers — their activations have wide dynamic range and are the most quantization-sensitive. If you lose more than a point, do per-channel weight quantization and consider QAT for just the first block. (Mixed precision and where to spend bits is covered in [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis).)
- **What if an op falls back to a generic kernel?** TinyEngine's whole advantage is that it generated specialized code; if your model uses an op the generator does not specialize, you lose both the speed and the tight memory plan for that op. Keep the op set inside what the engine supports — this is a real constraint on what architectures you can search.
- **What if patches make latency worse?** They will, by the recompute factor, on a compute-bound stage. On an MCU you are almost always memory-bound in the early stage so it is a win, but verify — measure latency before and after, do not assume.
- **What about residual connections across the patched boundary?** A residual add needs both operands live, which can pin a tensor across the patch loop and erode the memory saving. MCUNetV2 keeps the patched stage's topology simple precisely to avoid this; if you add cross-patch residuals you must account for their lifetime in the peak.
- **What if you need higher accuracy than ~65%?** Then the 256 KB MCU is the wrong device. There is a ceiling to what fits, and chasing the last points past it is exactly the "buy a bigger chip" case.

## Stress test: when the early stage still won't fit

The decision tree above tells you *whether* to reach for co-design. This section is the harder, more useful question: you reached for it, you applied every trick in the post, and the early-stage activation **still** won't fit — or the patch recompute has ballooned past the point where it pays. What actually breaks, in what order, and what do you do? Walk the failure path the way you would at 2 a.m. with a linker map that says "region RAM overflowed by 41 KB."

**Failure 1 — the per-tile peak is still over the ceiling even at a fine grid.** You patched the early stage, went from a $2\times2$ to a $3\times3$ to a $4\times4$ grid, and the high-water mark is still 30 KB over. Here is why cranking $n$ stops helping: the per-tile peak is $\frac{HWC}{n^2}$ *plus the halo*, and the halo does **not** shrink with $n$ — it is set by the receptive field. As $n$ grows, the clean term collapses but the halo term becomes a *larger fraction* of each shrinking tile, so the per-tile input window stops falling and eventually rises. There is a floor: $M_{\text{patch}} \gtrsim (\text{halo-padded tile}) \times C$, and below that floor more tiles only buy you recompute, not memory. If you have hit the floor, the grid is not your problem — the channel count $C$ at the patched stage is. The fix is to push the first stride-2 downsample *earlier* (so the patched stage runs at lower $C$ or smaller $H\times W$ before the expensive expansion), or to narrow the early width multiplier $w$, both of which attack $C$ and $HWC$ directly. This is TinyNAS's job, and if you are hand-designing, it is the moment to go back to the architecture, not the schedule.

**Failure 2 — recompute overhead has eaten the latency budget.** You got the peak under the ceiling, but to do it you went to a fine grid on a stage with a non-trivial receptive field, and the worked example's disaster came true: the input windows overlap so much that the early stage now does 2–3× its original FLOPs, and the frame time blew past your real-time deadline. Diagnose it precisely with the halo formula — compute $(t s + (r-s))^2/(t s)^2 - 1$ for your actual $t$, $s$, $r$; if it is north of ~40% you are in the danger zone. Two real fixes, in order of preference: (a) **redistribute receptive field** — move early downsampling/large-kernel ops out of the patched stage into the whole-map late stages, shrinking $r$ so the halo thins; this is the MCUNetV2 move and it usually recovers the budget. (b) **Patch fewer stages** — only the first one or two layers need patching to flatten the peak; patching deeper just stacks more receptive field into the halo. If neither gets the overhead under control, you have learned something important: this stage is genuinely compute-bound, not memory-bound, and patch inference is the wrong tool for a compute-bound stage. Reach for the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) to confirm which resource actually binds before you spend another day on tiling.

**Failure 3 — a residual connection pins a tensor across the patch loop.** You added a skip connection that spans the patched boundary, and suddenly the peak jumped back up. A residual add needs both operands live simultaneously; if one operand is the *whole* early-stage input and the other is being produced tile by tile, the whole input is pinned in SRAM for the entire patch loop, defeating the per-tile saving. The peak is now "one tile **plus** the full pinned skip tensor," which can be most of what you were trying to avoid. The fix is structural: keep the patched stage's topology a simple chain (no cross-boundary residuals), and place residual adds only inside a single tile's computation or after the patch loop reassembles the small output. MCUNetV2 keeps the patched stage deliberately plain for exactly this reason. If your architecture *needs* the cross-patch skip for accuracy, you have a genuine conflict between the memory schedule and the model — and that usually resolves toward "the 256 KB chip is the wrong device," which is failure 5.

**Failure 4 — int8 was not enough and the int32 accumulator leaked.** The model fits on paper at one byte per element, but the runtime is briefly holding an int32 intermediate at the early-stage resolution and your peak is 4× a tensor you thought was int8. This is the requantization-fusion bug from earlier: a generic runtime that does not fuse requant into the conv lands the int32 accumulator in SRAM. Confirm it by diffing the theoretical int8 peak against the measured peak — a ~4× gap on one tensor is the signature. The fix is to use an engine that fuses requantization (TinyEngine does; verify your runtime does too), or, failing that, to restructure so the accumulator never has to spill. There is no architecture change that helps here; it is purely an engine property, which is one more argument for the code-generating runtime.

**Failure 5 — you have exhausted the tricks and you are still over.** Every lever is pulled: int8 throughout, in-place depthwise, fusion, patch inference with redistributed receptive field, the narrowest width and lowest resolution that hits your accuracy floor — and the peak is still above the ceiling, or the accuracy at a feasible config is below what the product needs. This is not a failure of the technique; it is the technique correctly telling you the truth. The 256 KB chip cannot hold the model your task requires. The honest moves are, in order: (a) relax the *task* — can a cheaper proxy (visual wake words instead of full classification, a gated cascade where a tiny always-on model wakes the bigger one) meet the product need? (b) **buy the bigger chip** — stepping from 256 KB to 512 KB or 1 MB of SRAM is often a sub-dollar BOM change that dissolves the entire problem, and per the decision tree that is the right answer the moment the silicon delta is cheaper than the engineering delta. (c) Move the inference off the device entirely if connectivity and latency allow. The skill being tested in this failure is knowing when to stop optimizing — co-design has a ceiling, and pretending it does not is how you burn a quarter chasing 30 KB that a fifty-cent part would have given you for free.

The through-line across all five: each failure has a *specific signature* in the numbers (peak floor independent of $n$; halo overhead from the formula; a peak jump when a skip is added; a 4× gap from an int32 leak; a feasible-config accuracy below target), and each has a *specific fix at a specific layer* of the stack (architecture, schedule, topology, engine, or hardware). The reason the post spent so long on the math is that the math is the diagnostic. When the linker map overflows, you do not guess — you compute the per-tile peak, the halo overhead, and the pinned-tensor bytes, and the number that is wrong tells you which knob to turn.

## Case studies and the broader pattern

The MCUNet line is the flagship, but it sits in a small family of work that all says the same thing — co-design beats single-axis optimization at the extreme low end.

**MicroNets (Banbury et al., 2021)** studied what actually predicts latency and energy on MCUs and found, usefully, that for these tiny models running on these runtimes, **latency is roughly linear in the number of operations** and energy tracks it closely — so a FLOP-aware (really, op-aware) NAS objective is a good proxy. They used differentiable NAS to find models on the accuracy–latency–memory frontier for MCUs across the standard tinyML tasks (visual wake words, keyword spotting, anomaly detection). The takeaway that complements MCUNet: once you respect the memory wall, ops are a decent stand-in for latency on MCUs, which is *not* generally true on phones or GPUs (see [efficientnet, shufflenet, and the FLOPs-latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap) for why the proxy breaks down on richer hardware).

**TinyEngine as a standalone result.** Even without the search, the engine alone is instructive. The MCUNet paper's ablation — same model, TFLM versus TinyEngine — shows the runtime closing a large memory and latency gap by itself. The lesson for any edge engineer: before you redesign the model, check whether your *runtime* is wasting the budget. A code-generating or graph-optimizing runtime can be the cheapest win available, and it requires no retraining. This is the runtime half of the optimization story covered across [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization) and [inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared).

**Patch-based / fused execution beyond MCUs.** The patch idea — never materialize a giant intermediate, process it in tiles — is the same family as fused attention (FlashAttention tiles the attention matrix so it never lands in HBM) and as tiled convolution in GPU kernels. The shapes differ but the principle is identical: when a tensor is too big to hold, compute it region by region and pay a little recompute to save a lot of memory. Recognizing this pattern is what lets you transfer the trick to new settings.

**Visual wake words in production.** The most shipped real-world use of this line of work is always-on person/object detection on battery-powered sensors — smart doorbells, occupancy sensors, wildlife cameras. The combination of sub-300 KB SRAM, sub-1 MB flash, and >90% wake-word accuracy is what makes a multi-month coin-cell vision sensor possible. That is the genuine payoff: not a leaderboard point, but a class of product that did not exist before the memory wall was beaten.

## Key takeaways

- **Peak activation memory, not parameters or flash, is the wall on a microcontroller**, and it is front-loaded into the early, high-resolution layers because feature-map area shrinks quadratically with depth while channels grow only linearly.
- **MCUNet is a system, not a model**: TinyNAS (budget-aware search over architecture, resolution, and width) plus TinyEngine (a code-generating runtime with in-place depthwise, fusion, and a model-specific memory plan), tuned against the same SRAM/flash budget.
- **Input resolution and channel width are the two highest-leverage knobs** on early-stage memory, which is why TinyNAS searches them as first-class dimensions rather than fixing them.
- **TinyNAS optimizes the search space first** (pick the (resolution, width) family whose feasible models pack the most FLOPs into the budget via the FLOPs CDF), then searches inside it — so every sampled subnet already fits.
- **Patch-based inference cuts the early-stage peak by roughly $n^2$** (4–8× realized) by processing the high-resolution stage one spatial tile at a time, so the giant feature map never fully materializes — and that headroom buys back the input resolution that recovers accuracy.
- **The recompute cost of patches is the receptive-field halo**; MCUNetV2 minimizes it by redistributing receptive field out of the early stage and into the cheap late stages, holding overhead to ~10–20%.
- **Swapping the runtime can be the cheapest win** — same model, TFLM → TinyEngine, fits where it did not and runs ~3× faster, with no retraining.
- **Co-design pays off only at the sub-megabyte extreme**; when you have tens of MB or can step up the chip cheaply, standard quantization plus a good runtime is the right, far cheaper answer.
- **Measure peak SRAM from the linker map and a runtime high-water mark**, latency after warm-up at batch 1, and accuracy on the quantized deployed graph — never from a spreadsheet.

## Further reading

- Lin, Chen, Lin, Gan, Han. *MCUNet: Tiny Deep Learning on IoT Devices.* NeurIPS 2020 — the original TinyNAS + TinyEngine co-design and the 256 KB ImageNet result.
- Lin, Chen, Lin, Gan, Han. *MCUNetV2: Memory-Efficient Patch-Based Inference for Tiny Deep Learning.* NeurIPS 2021 — patch-based inference and receptive-field redistribution.
- Banbury et al. *MicroNets: Neural Network Architectures for Deploying TinyML Applications on Commodity Microcontrollers.* MLSys 2021 — the op-count latency proxy and MCU NAS across tinyML tasks.
- The MIT Han Lab repos: `mit-han-lab/mcunet` (models and TinyNAS) and `mit-han-lab/tinyengine` (the code-generating runtime).
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame; [TinyML on microcontrollers](/blog/machine-learning/edge-ai/tinyml-on-microcontrollers) for the device baseline; [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) for activation lifetimes and arena planning; [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas) for the latency-objective version of budget-aware search; [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for honest measurement; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how to sequence all of this on a real project.
