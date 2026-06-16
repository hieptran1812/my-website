---
title: "The edge hardware landscape: from Cortex-M to mobile NPUs to Jetson"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Learn to read the silicon before you touch the model: compute, memory, bandwidth, and power across every edge tier, so your next optimization speeds the model up instead of slowing it down."
tags:
  [
    "edge-ai",
    "model-optimization",
    "hardware",
    "npu",
    "quantization",
    "inference",
    "efficient-ml",
    "tinyml",
    "jetson",
    "roofline",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-edge-hardware-landscape-1.png"
---

A few years ago I watched a very good engineer make a model 30% slower by optimizing it. The task was real-time pose estimation on a mid-range Android phone. The model was a tidy little convolutional network, and our engineer did exactly what every textbook says: he profiled the FLOPs, found that the depthwise-separable blocks at the tail were eating most of the multiply-accumulate operations, and replaced several of them with a clever factorization that cut the FLOP count by about 22%. The math was correct. The FLOP counter agreed. He shipped it to the test farm. And the p50 latency went *up* — from 41 ms to 53 ms.

The reason was not subtle once we looked at the right number. His "cheaper" blocks had more, smaller tensors flowing between them. On that phone's GPU, the bottleneck was never the arithmetic; it was the rate at which the chip could stream activation tensors in and out of off-chip DRAM. He had traded a handful of multiplies — which the GPU had in abundance — for extra round trips to memory, which the GPU was already starved for. He optimized the resource that was plentiful and spent more of the resource that was scarce. The model was *memory-bandwidth-bound*, and he had treated it as *compute-bound*. Every assumption in his head was about a machine that did not exist.

This post is about not making that mistake. The central claim of the whole "Optimizing AI Models for the Edge" series is that you choose techniques — quantization, pruning, distillation, efficient architectures — by reading them off an accuracy–efficiency Pareto frontier. But that frontier is *defined by the hardware*. The same model is a different optimization problem on a Cortex-M7 than on a Jetson Orin than on an iPhone's Neural Engine, because the ratio of compute to memory bandwidth, the precision the silicon loves, and the watts you are allowed to burn are wildly different across those targets. You cannot optimize a model without knowing the silicon it runs on. Figure 1 sketches the tiers we will walk through, and the six orders of magnitude they span.

![A tree diagram of edge hardware tiers from microcontrollers through mobile CPUs, GPUs, and NPUs to Jetson-class boards and custom FPGA or ASIC silicon, each annotated with its compute and power range](/imgs/blogs/the-edge-hardware-landscape-1.png)

By the end of this post you will be able to look at a target device, find its three numbers that actually matter — peak compute, memory capacity, and memory bandwidth — predict whether a given model will be compute-bound or memory-bound on it, know which numeric precision the chip wants, and know how to *query* all of this from a shell or a Python session instead of guessing. That is the foundation everything else in the series builds on. Let us get the vocabulary and the numbers straight first, then do the physics, then write the code that reads it off a live device.

## 1. The three numbers that decide everything

Before we tour the tiers, fix the mental furniture. For inference, a device is characterized — to first order — by exactly three numbers, plus a power budget:

- **Peak compute**: how many multiply-accumulate operations per second the chip can do, usually quoted as GFLOP/s (billion floating-point ops per second) or, for integer accelerators, **TOPS** (trillion operations per second, almost always meaning int8). One MAC is two operations (a multiply and an add), so a number like "275 TOPS" means roughly $1.4 \times 10^{14}$ int8 MACs per second at peak.
- **Memory capacity**: how many bytes you have to hold weights plus activations plus the runtime. On an MCU this is KB of SRAM; on a phone it is shared GB of LPDDR; on a Jetson AGX it is up to 64 GB. If your model's working set does not fit, you either stream from slower storage (slow) or you do not run.
- **Memory bandwidth**: how many bytes per second you can move between the compute units and main memory, in GB/s. This is the number people forget, and it is the one that bit our pose-estimation engineer. A chip can have enormous peak compute and still crawl because it cannot feed the math units fast enough.
- **Power envelope**: how many watts you are allowed to dissipate before you drain the battery, melt the enclosure, or trip thermal throttling. This is a hard wall on the edge in a way it never is in a datacenter.

The single most useful derived quantity is **arithmetic intensity**: the number of compute operations you do per byte of memory traffic, in FLOP/byte (or OP/byte for integer). A matrix multiply that reads a weight once and does one MAC with it has low intensity; a convolution that reuses each weight across a whole feature map has high intensity. We will make this precise in the science block, because it is the lever that decides compute-bound versus memory-bound — the thing the FLOP counter is blind to. The roofline model, which I cover in depth in [the-roofline-model-where-your-bottleneck-lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), turns these four numbers into a single picture that tells you which wall you are hitting.

Keep those four numbers in mind as a checklist. For every tier below I will give you real values for all of them. When you internalize that, say, a phone NPU might offer 26 TOPS but only 40-something GB/s of usable bandwidth from shared LPDDR, you will start to *feel* when a model is going to be starved instead of finding out on the test farm.

## 2. Tier 0 — microcontrollers: the TinyML floor

Start at the bottom, because the constraints there are so brutal that they clarify everything above. A microcontroller (MCU) is a complete computer — CPU, memory, peripherals — on one cheap chip, designed to run on a coin cell for months. This is the land of TinyML: keyword spotting on a smart speaker, anomaly detection on a motor, person-detection on a doorbell, all running on a chip that costs a couple of dollars and sips milliwatts.

The numbers are humbling. An ARM **Cortex-M0+** runs at tens of MHz, has no floating-point unit at all, and might give you 16–64 KB of SRAM and 256 KB of flash. A **Cortex-M4** adds a single-precision FPU and a DSP extension and runs around 80–180 MHz. A **Cortex-M7**, the high end of classic microcontrollers, runs up to ~480–600 MHz, has a double-precision FPU on some variants, a 6-stage superscalar pipeline, and on a chip like the STM32H7 you get around 1 MB of SRAM and up to 2 MB of internal flash. The popular **ESP32** family (Xtensa or RISC-V cores around 160–240 MHz, ~520 KB SRAM, Wi-Fi/BLE built in) sits in similar territory and dominates hobbyist and IoT deployments.

What does that buy you in compute? A Cortex-M7 at 480 MHz, even with the SIMD-ish DSP instructions that let it do a couple of 8-bit MACs per cycle, lands somewhere around a few hundred million MACs per second — call it well under 1 GFLOP/s of *useful* throughput on real kernels. There is no DRAM in the usual sense: your "memory bandwidth" is the SRAM, which is fast and on-chip but measured in *kilobytes*. Power is in the **single-digit to tens of milliwatts** range when active, microwatts when sleeping. Price is a dollar or two for the chip.

The consequence is that on an MCU, *everything* is a memory problem. You do not have room for fp32 weights; you barely have room for int8 weights. A model that is 5 MB on disk simply cannot live in 1 MB of SRAM — its weights must sit in flash and be read as needed, and you must arrange the computation so the *activations* (the intermediate tensors) fit in a fixed scratch buffer called the **tensor arena**. The frameworks here — **TensorFlow Lite for Microcontrollers** (TFLM) and ARM's **CMSIS-NN** kernel library — are built around this: no dynamic memory allocation at all, a single pre-sized arena, int8 everywhere, and hand-tuned kernels that exploit the Cortex-M DSP instructions.

The landmark result that defined the field is **MCUNet** (Lin et al., NeurIPS 2020), which co-designed a tiny network (TinyNAS) with a memory-aware inference engine (TinyEngine) to run ImageNet-scale classification on a Cortex-M7 with 320 KB of SRAM and 1 MB of flash, reaching around 70% ImageNet top-1 — something people had assumed needed far more silicon. The lesson MCUNet taught, and the lesson the **MLPerf Tiny** benchmark suite institutionalized, is that on microcontrollers the binding constraint is *peak SRAM usage of the activation graph*, not parameter count and not FLOPs. You can have a model with few parameters that still blows the SRAM budget because one intermediate tensor is too wide.

#### Worked example: does a 5 MB int8 CNN fit a Cortex-M7?

Suppose you have an int8 image classifier with 5 million weights, so the weight storage is 5 MB. Take an STM32H743: 1 MB SRAM, 2 MB internal flash. The 5 MB of weights does **not** fit in flash either — you would need external QSPI flash (XSPI/OctoSPI), which is common and gives you tens of MB but at lower read bandwidth (think tens of MB/s, executed in place or DMA'd in). So the weights can live in external flash. The real question is the **activations**: if the model's peak activation footprint — the largest sum of live intermediate tensors at any one layer — is, say, 360 KB, then it fits comfortably in the 1 MB SRAM arena with room for the runtime. If instead an early layer produces a $112 \times 112 \times 32$ int8 feature map, that single tensor is $112 \cdot 112 \cdot 32 = 401{,}408$ bytes ≈ 392 KB, and with the input and the next layer's output live simultaneously you can blow past 800 KB and run out of arena. The fix is architectural (patch-based or strided early downsampling, as MCUNet's TinyEngine does), *not* quantization — you are already at int8. This is the canonical TinyML trap: a small model that does not fit because of one fat activation. Knowing the SRAM number up front is what tells you that before you waste a week.

## 3. Tier 1 — mobile CPUs: ARM big.LITTLE and NEON

Climb one rung and you hit the application processor inside every phone: a multi-core ARM CPU, almost always in a **big.LITTLE** (now "DynamIQ") arrangement — a cluster of small, power-efficient cores (e.g. Cortex-A55/A510) paired with a cluster of big, fast cores (Cortex-A78/A715/X-series), plus sometimes a "prime" core. The scheduler migrates threads between clusters to balance performance and battery. A flagship phone CPU runs the big cores around 2.8–3.3 GHz.

The number that matters for ML on these cores is **NEON**, ARM's SIMD (single-instruction-multiple-data) extension. A NEON unit has 128-bit vector registers, which means one instruction can operate on 4 fp32 lanes, 8 fp16 lanes, or 16 int8 lanes at once. Newer cores add **dot-product** instructions (`SDOT`/`UDOT`) that do a 4-element int8 dot product into an int32 accumulator in a single instruction, and ARMv8.2 adds native fp16 arithmetic. This is why a well-written int8 kernel on a phone CPU can be 2–4× the throughput of the fp32 path on the *same* core — you are pushing 16 int8 lanes instead of 4 fp32 lanes through the same register file, plus the dot-product fast path. We will make the lane-counting argument rigorous in the SIMD section.

Concretely: a single big core doing int8 GEMM with NEON dot-product might sustain a few GMAC/s; the whole multi-core cluster with a good threaded kernel (XNNPACK, the engine under TFLite's CPU path, or oneDNN/ruy) can reach low tens of GOP/s int8. Memory is the **shared system LPDDR** — on a flagship, LPDDR5/5X giving real-world bandwidth in the range of 40–60 GB/s shared across the whole SoC, with maybe single-digit MB of last-level cache. Power for a sustained ML workload on the big cores is **2–5 W**, which on a phone means you will thermally throttle within tens of seconds if you peg it.

The CPU is the universal fallback: every op is supported, shapes can be dynamic, and there are no driver surprises. It is also the *slowest and most power-hungry* per inference of the accelerated options. The art on mobile is using the CPU for the glue and the unsupported ops while pushing the heavy matmuls to the GPU or NPU — which sets up the single most important hazard on mobile, the op-fallback cliff, in §6.

## 4. Tier 2 — mobile GPUs, and the GPU-vs-NPU choice

Every mobile SoC ships a GPU: Qualcomm **Adreno**, ARM **Mali**, Imagination **PowerVR**, and Apple's in-house GPU. For graphics these are tile-based renderers; for ML they are general SIMT (single-instruction-multiple-thread) machines you program through compute APIs — **OpenCL**, **Vulkan compute**, or Apple's **Metal**. TFLite's **GPU delegate** and Core ML both target them.

A flagship mobile GPU offers on the order of **1–3 TFLOP/s of fp16** (a high-end Adreno 7-series or Apple GPU is in this band; fp16 is the native ML precision on mobile GPUs, and fp32 is roughly half-rate). They share the same LPDDR as the CPU — there is **no dedicated VRAM** on a phone, which is the crucial architectural fact. So your memory bandwidth is the same 40–60 GB/s shared pool, and now you are also competing with the display and the CPU for it. Power for sustained GPU compute is **2–4 W**.

So when do you pick the GPU over the NPU? The honest answer:

- **GPU wins** when your model has ops the NPU does not support, when you need fp16 precision (some models lose too much at int8), when shapes are dynamic (NPUs hate dynamic shapes — more on that in §6), or when you want one code path across many phones (the GPU delegate is far more portable than the fragmented NPU APIs). The GPU is the *flexible* accelerator.
- **NPU wins** when your model is a clean, static, int8-friendly graph of standard ops — then it is several times faster and *much* more power-efficient than the GPU. The NPU is the *specialized* accelerator.

A useful intuition: the GPU is a Swiss-army knife that does ML acceptably; the NPU is a scalpel that does *one kind* of ML extremely well and is useless outside its lane. Most production mobile ML I have shipped uses the NPU when the model cooperates and falls back to the GPU (not the CPU) when it does not, because the GPU is a much gentler fallback. Figure 4 quantifies the per-tier gap on a single model.

![A two-column before-and-after figure contrasting the same MobileNet running on a mobile CPU in floating point versus on a mobile NPU in int8, showing roughly an eight times latency reduction and a large power reduction at a small accuracy cost](/imgs/blogs/the-edge-hardware-landscape-4.png)

## 5. Tier 3 — mobile NPUs and the int8-first world

The NPU (Neural Processing Unit) — also called an NPU, TPU, APU, or "AI engine" depending on the vendor — is a dedicated matrix-multiply accelerator built into the SoC. The big four on phones:

- **Apple Neural Engine (ANE)**: 16 cores on recent A- and M-series chips, advertised around **15.8–35 TOPS** depending on generation (the A17 Pro's ANE is ~35 TOPS). You reach it only through Core ML; you cannot program it directly. It is fp16/int8 and is astonishingly power-efficient when your model maps to it.
- **Qualcomm Hexagon** (the "AI Engine" inside Snapdragon): a DSP-derived tensor accelerator. A Snapdragon 8-series Hexagon is quoted in the **tens of TOPS** range. Reached via the QNN SDK, the NNAPI Hexagon delegate (on older Android), or TFLite's delegates.
- **Google Edge TPU** (Coral, and the Tensor SoC's TPU): the Coral Edge TPU does **4 TOPS int8** at about 2 W (so ~2 TOPS/W), and runs **int8-only** models compiled by the Edge TPU compiler. Anything it cannot map runs on the host CPU.
- **MediaTek APU**: the NPU in Dimensity SoCs, again tens of TOPS, reached via NeuroPilot / NNAPI.

Three things define how you must treat an NPU, and they are the same across vendors:

1. **They are int8-first.** The MAC arrays are built for 8-bit integer multiply with 32-bit accumulation. Many support fp16 and a growing number support int4 (great for LLM weights) or other low-bit modes, but int8 is the native, full-rate path. If your model is fp32, the NPU either refuses it or silently does worse. This is the hard coupling between *hardware* and *quantization* that the whole quantization track of this series exists to exploit — see [a-taxonomy-of-model-compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for where quantization sits among the four levers.
2. **They hate dynamic shapes.** An NPU compiler ahead-of-time plans the entire dataflow — which tile goes to which PE, when each weight is loaded — for a *fixed* tensor shape. A dynamic dimension (variable sequence length, variable batch, a shape that depends on the data) breaks that planning. Some runtimes handle it by padding to a max shape (wasteful) or by recompiling (a latency cliff); many just reject the model. This is why on-device LLMs do so much work to make the sequence dimension static or bucketed.
3. **They support a limited op set.** The NPU implements a curated list of ops. Hit one it does not implement and the runtime splits your graph and runs that piece on the CPU — the fallback cliff of §6.

Why do NPUs love int8 and reach 2–4× their fp16 rate? Two reasons, both physical, and figure 7 lays them side by side. First, **lanes per register**: an 8-bit value is half the width of fp16, so a fixed-width datapath holds twice as many int8 operands, and a MAC array with a fixed transistor budget can pack roughly twice as many int8 MAC units as fp16 ones. Second, **memory traffic**: an int8 weight is one byte versus two for fp16, so for the same bandwidth you stream twice as many weights per second — and since NPUs are often bandwidth-limited at the edges of the array, halving the byte count directly raises sustained throughput. The arithmetic itself is also cheaper: an 8-bit integer multiplier is far smaller and lower-energy than a floating-point one (we will see the energy figures in §7).

![A two-column before-and-after figure comparing the fp16 datapath against the int8 datapath on the same accelerator die, showing that int8 fits twice as many lanes per register and halves the bytes per weight to yield two to four times the throughput](/imgs/blogs/the-edge-hardware-landscape-7.png)

There is also a sizing subtlety that trips people up: the advertised TOPS of a mobile NPU is a *peak* int8 number, and you will almost never see it in practice for the same reason the §7 machine-balance table predicted — a 26-TOPS NPU hanging off ~50 GB/s of shared LPDDR has a machine balance around 520 OP/byte, so unless your model reuses each byte hundreds of times you are bandwidth-limited and running at a fraction of peak. Two NPUs with identical TOPS can deliver very different real latency if one has more on-chip SRAM (more reuse stays local) or a wider path to LPDDR. When you compare phone NPUs, weight the on-chip-memory and bandwidth specs at least as heavily as the TOPS headline; the TOPS is the ceiling, the memory system is the floor, and real models live near the floor.

A short word on **DSPs**, because they blur into NPUs. A digital signal processor (e.g. the Hexagon DSP before it grew tensor units, or the DSP blocks in many SoCs) is a programmable core specialized for the multiply-accumulate-heavy inner loops of signal processing, with wide vector units (Hexagon's HVX is 1024-bit) and very low power. Historically the mobile ML accelerator *was* the DSP; modern NPUs are often DSPs with bolted-on systolic tensor arrays. For the optimizer, a DSP behaves like a low-power, int-friendly accelerator with a more flexible (programmable) op model than a fixed-function NPU but lower peak throughput — which makes it the natural home for the int8 signal-processing front-ends (audio feature extraction, FFTs) that sit *before* a neural net, often on the same chip as the tensor NPU that runs the net itself.

## 6. The fallback cliff — the most expensive surprise on the edge

Here is the failure mode that has cost me more debugging hours than any other on mobile, and it deserves its own section. You quantize a model to int8, you target the NPU, you measure, and the latency is *worse* than the plain CPU build. Nothing in the FLOP count or the model size predicted it. The cause is almost always a **partition split**: one op in your graph is not supported by the NPU, so the runtime cuts the graph at that op, runs the unsupported piece on the CPU, and ships tensors back and forth across the SoC fabric to do it. Figure 5 shows the shape of the hazard.

![A dataflow graph showing an input tensor entering an NPU convolution, a delegate routing decision that either keeps the op on the NPU or copies the tensor to the CPU for a fallback op that is far slower, with both paths rejoining at the output](/imgs/blogs/the-edge-hardware-landscape-5.png)

Why is the round trip so expensive? Three costs stack up. First, the **data transfer**: an activation tensor that lived in the NPU's local memory must be copied to a buffer the CPU can read, which on shared-memory SoCs is "free" in bytes but costs a cache flush and a synchronization barrier. Second, the **synchronization stall**: the NPU must finish and signal, the CPU must wake and run, then hand back — you have serialized two engines that wanted to pipeline. Third, the **fallback op itself** runs on the slowest engine, often in a precision (fp32) that forces a dequantize-then-requantize around it. The net effect is that a single unsupported op in the middle of a graph can turn a 3 ms NPU inference into a 30 ms one, because you paid the NPU↔CPU handoff *twice* (out to CPU, back to NPU) around it.

The ops that bite most often: exotic activations the NPU lacks (some GELU/SiLU variants, custom layers), dynamic-shape ops (`NonMaxSuppression` in detectors, anything with data-dependent output size), unusual layouts or transposes the NPU compiler will not fuse, and "control flow" ops. The diagnosis is always the same: enumerate the delegate's partition plan and look for how many partitions it created. One partition that covers the whole graph is the dream; ten partitions means ten handoffs.

How do you fix it once you find it? In order of preference: (1) **replace the op** with a supported equivalent (swap an exotic activation for ReLU6, refactor a dynamic op to a static one) — this is where knowing the hardware shapes the *model architecture*, the bridge to the efficient-architecture lever; (2) **make shapes static** by fixing or bucketing the dynamic dimension; (3) **move the boundary** so the fallback happens once at the very end (e.g. do post-processing entirely on CPU after a single NPU pass) instead of mid-graph; (4) accept the GPU as the fallback instead of the CPU. The general principle — that an "unsupported op" is a hardware-fit problem you solve at the model level — is exactly why the quantization posts and the architecture posts in this series keep pointing back at the silicon.

### Quantization-hardware fit: precision is a hardware contract, not a knob

It is worth being precise about *which* precisions each accelerator actually runs at full rate, because this is the single tightest coupling between the hardware and the model-compression levers, and it is where the most accuracy gets needlessly thrown away. The mental model to install is that a precision is not a dial you turn for "more speed" — it is a **contract with the silicon**: the chip runs one or two precisions at full throughput, tolerates a couple more at reduced rate, and refuses the rest.

- **int8 with int32 accumulation** is the universal full-rate path on every NPU, the Edge TPU, the Hexagon tensor unit, the MediaTek APU, and the Cortex-M DSP. It is the default you should target unless you have a reason not to. The accumulation is *always* wider than the operands (int32 from int8 inputs) so that a long reduction does not overflow — a detail the quantization track makes rigorous, but the hardware fact is that the accumulator is fixed-width and you must keep partial sums inside it.
- **fp16 (and bf16)** is the GPU's native ML precision and is supported on most NPUs at roughly half the int8 rate (because it is twice the data width — the §8 lane argument run backward). Reach for it when a model loses unacceptable accuracy at int8 (some attention layers, some normalization-heavy networks) and you can afford the throughput hit.
- **int4 / lower** is the newest frontier and is *unevenly* supported: recent NPUs and GPUs add int4 weight paths specifically for LLM weight quantization, where the §7 memory-bound argument makes 4-bit weights a near-free speedup for decode. But int4 is rarely a full-rate compute mode — it is usually a *storage and bandwidth* win where weights are dequantized to int8 or fp16 just before the MAC. Knowing whether your chip does true int4 MACs or merely int4 storage changes whether you expect a compute speedup or only a bandwidth one.
- **fp32** is the CPU/GPU fallback and almost never the accelerator's fast path. If your deployed model is still fp32, you have left the entire accelerator on the table.

The practical rule that follows: **decide your target precision from the chip's full-rate contract first, then quantize to hit it** — not the other way around. Targeting an Edge TPU means int8-only, period; targeting an Apple Neural Engine means fp16 or int8; targeting a Jetson means you have the luxury of int8 *and* fp16 via TensorRT and can mix them per layer. This is exactly the bridge into the compression levers: the taxonomy in [a-taxonomy-of-model-compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) treats quantization as one of four levers, but the *bit-width you pick* is dictated by the silicon's contract, and getting it wrong (shipping fp32 to an int8 chip, or forcing int4 where there is no int4 path) is the most common way teams either leave performance unclaimed or chase a speedup the hardware was never going to give.

One more subtle coupling: **per-tensor versus per-channel quantization** is also a hardware-fit question. Per-channel weight quantization (a separate scale per output channel) recovers accuracy almost for free in software, but some fixed-function accelerators only implement per-tensor scaling in their fast path and fall back (or refuse) on per-channel. So the *form* of your quantization, not just the bit-width, has to match what the chip's MAC array and requantization unit actually support. Always check the accelerator's quantization spec — it tells you per-tensor vs per-channel, symmetric vs asymmetric, and which activation ranges the requantizer can represent — before you assume your software-optimal quantization will run on-device.

## 7. The science block: the memory hierarchy and the bandwidth wall

Now the physics, because it explains every "surprise" above. The reason data movement, not arithmetic, dominates both energy and often latency is captured in a famous set of numbers from Mark Horowitz's 2014 ISSCC keynote, "Computing's Energy Problem (and what we can do about it)." For a 45 nm process, the *energy per operation* looks roughly like this:

| Operation | Approx. energy | Relative to a MAC |
| --- | --- | --- |
| 8-bit integer add | ~0.03 pJ | ~0.2× |
| 8-bit integer multiply | ~0.2 pJ | 1× |
| 16-bit float multiply | ~1.1 pJ | ~5× |
| 32-bit float multiply | ~3.7 pJ | ~18× |
| Register file read | ~0.1 pJ | <1× |
| On-chip SRAM access (8–32 KB) | ~5 pJ | ~25× |
| Off-chip DRAM access | ~640 pJ | ~3000× |

(These are order-of-magnitude figures from Horowitz's data, scaled to a MAC; treat them as *approximate* — the exact numbers move with process node and memory size, but the *ratios* are robust and that is what matters.) Figure 2 stacks them so the cliff between on-chip and off-chip is visible at a glance.

![A vertical stack diagram of the memory hierarchy from registers and the MAC unit through on-chip SRAM down to off-chip DRAM and flash, annotated with the energy per access rising from fractions of a picojoule to hundreds of picojoules](/imgs/blogs/the-edge-hardware-landscape-2.png)

Read the table again and let the ratio land: **a single off-chip DRAM access costs roughly 3000× the energy of the int8 multiply it feeds.** A weight that lives in DRAM costs ~640 pJ to fetch and then ~0.2 pJ to use. So if you read each weight from DRAM and use it once, **99.97% of your energy went to moving data, not computing.** This is why every accelerator is, underneath, a machine for *avoiding* DRAM access — caches, on-chip SRAM scratchpads, weight-stationary dataflows, and tiling all exist to amortize that 640 pJ across as many cheap MACs as possible.

We can make "as many as possible" precise. Define **arithmetic intensity** $I$ as operations per byte of memory traffic:

$$I = \frac{\text{number of MACs}}{\text{bytes moved to/from main memory}} \quad \left[\frac{\text{OP}}{\text{byte}}\right]$$

A device has a peak compute rate $P$ (OP/s) and a peak memory bandwidth $B$ (byte/s). The roofline insight is that the achievable rate is bounded by *both*:

$$\text{achievable OP/s} \le \min\big(P,\; I \cdot B\big)$$

When $I \cdot B < P$ — that is, when $I < P/B$ — you are **memory-bound**: you cannot feed the math units fast enough, and adding more compute (or cutting FLOPs!) does nothing. When $I > P/B$ you are **compute-bound** and cutting FLOPs helps. The crossover ratio $P/B$ is the **machine balance**, in OP/byte. This single inequality is what our pose-estimation engineer needed. His phone's GPU had, say, $P \approx 1.5$ TFLOP/s and $B \approx 50$ GB/s, so its machine balance was $P/B \approx 30$ FLOP/byte. His model's intensity was below that — it was memory-bound — so cutting 22% of the FLOPs (lowering an already-irrelevant numerator) while *raising* memory traffic (more, smaller tensors → more bytes moved) pushed him further into the memory-bound regime and made it slower. The roofline I derive fully in [the-roofline-model-where-your-bottleneck-lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) turns this inequality into the picture you should draw before every optimization.

Here is the practically vital corollary for the edge: **quantization helps memory-bound models even if it does not change the FLOP count.** Going from fp16 to int8 halves the bytes per weight and per activation, which *doubles* the arithmetic intensity $I$ (same MACs, half the bytes), which on a memory-bound model directly raises achievable throughput — *before* you count any speedup from faster int8 math. This is why int8 quantization so reliably speeds up LLM decoding on the edge: autoregressive decode is brutally memory-bound (you stream the entire weight matrix to produce one token), so halving the weight bytes nearly doubles tokens/s. The science says cut bytes, not FLOPs, when you are below the machine balance.

Let us nail down the machine balance with concrete numbers per tier, because the crossover ratio $P/B$ is the one number that tells you, for *that* device, how much reuse a kernel needs before it stops being memory-bound. Compute it as peak-OP/s divided by peak-byte/s:

| Tier | Peak compute $P$ | Bandwidth $B$ | Machine balance $P/B$ |
| --- | --- | --- | --- |
| Mobile CPU (int8) | ~30 GOP/s | ~50 GB/s | ~0.6 OP/byte |
| Mobile GPU (fp16) | ~1.5 TFLOP/s | ~50 GB/s | ~30 FLOP/byte |
| Mobile NPU (int8) | ~26 TOPS | ~50 GB/s | ~520 OP/byte |
| Jetson AGX (int8) | ~275 TOPS | ~204 GB/s | ~1350 OP/byte |

Read down that last column and a striking pattern appears: **the more compute a tier piles on, the higher its machine balance, so the harder it is to keep fed.** A mobile CPU is "balanced" at under 1 OP/byte — almost any kernel keeps it busy. An NPU at ~520 OP/byte demands that every weight be reused *hundreds* of times before the math units stop starving — which is exactly why NPUs are built as systolic arrays (§8) whose entire purpose is to manufacture that reuse, and exactly why a low-reuse op (an elementwise activation, a transpose, a tiny matmul) wastes an NPU even when it "fits." A figure with high FLOPs but low arithmetic intensity will run *below peak* on every high-balance accelerator; this is the formal statement of why FLOP-counting misleads on the edge.

We can also write the bandwidth-bound latency floor in closed form, generalizing the §7 LLM example to any layer. If a layer moves $W$ bytes of weights and $A$ bytes of activations (in and out) to and from main memory, its latency cannot beat

$$t_{\min} = \frac{W + A}{B}$$

regardless of how much compute the chip has. For a fully-connected layer with an $m \times n$ weight matrix at $b$ bytes per weight, $W = m n b$, and at batch 1 the activation traffic $A = (m + n) b$ is usually negligible next to $W$, so $t_{\min} \approx m n b / B$. Halving $b$ (int8 → int4, or fp16 → int8) halves this floor *directly*. That single equation is the whole reason the LLM world quantizes weights so aggressively, and it is why the right question for a memory-bound layer is never "how do I do fewer multiplies?" but "how do I move fewer bytes?" — fewer bits per weight, weight sharing, or keeping the working set on-chip so it never crosses the $B$ boundary at all.

#### Worked example: is on-device LLM decode compute- or memory-bound?

Take a 7-billion-parameter LLM quantized to 4-bit weights — about 3.5–4 GB of weights. On a laptop-class integrated setup with, say, $B \approx 100$ GB/s of effective bandwidth, generating one token in the decode phase requires reading essentially all the weights once (plus the KV cache). Reading 4 GB at 100 GB/s takes $4/100 = 0.04$ s = 40 ms — a *floor* of ~25 tokens/s set purely by bandwidth, before any compute. The compute is about $2 \times 7\text{e}9 = 1.4 \times 10^{10}$ FLOP per token; even a modest 1 TFLOP/s would do that in 14 ms, less than the 40 ms memory time. So decode is **memory-bound**: the chip finishes the math and waits for weights. The model's arithmetic intensity per token is roughly $1.4\text{e}10 \text{ MAC} / 4\text{e}9 \text{ byte} \approx 3.5$ OP/byte, far below any modern machine balance. The actionable consequence: to speed up decode you cut *bytes* (lower-bit quantization, smaller KV cache) — cutting FLOPs is useless. This is the entire reason 4-bit and even 2-bit weight quantization is standard for on-device LLMs, and it is a hardware fact, not a software preference.

## 8. SIMD, warps, and systolic arrays: how the matmul actually happens

We keep saying "int8 is 2–4× faster on the same die." Let us see the three mechanisms that make it true, because they are different at each tier.

**On the CPU (SIMD / NEON).** A 128-bit NEON register is a fixed bucket of bits. Fill it with fp32 and you get 4 lanes; with fp16, 8 lanes; with int8, 16 lanes. One vector instruction processes the whole register, so int8 does 16 MACs where fp32 does 4 — a 4× lane advantage. ARM's `SDOT`/`UDOT` int8 dot-product instruction goes further: it does a 4-way int8 multiply-and-accumulate into int32 in a single instruction, so a 128-bit register holding 16 int8 values produces 4 int32 dot-products per instruction. This is pure data-width arithmetic: narrower data → more lanes → more work per instruction at the same clock.

Make the lane count a formula. For a register of width $R$ bits and a datatype of width $w$ bits, the number of lanes is $L = R / w$, and the peak MAC throughput per core is $L$ times the issue rate. So going from fp32 ($w=32$, $L=4$) to int8 ($w=8$, $L=16$) is a $4\times$ lane factor; the dot-product instruction then folds 4 multiply-adds into one issue slot, and the newest **`i8mm`** (int8 matrix-multiply) extension does an even wider int8 outer-product per instruction. This is why the §11 trick of grepping `/proc/cpuinfo` for `asimddp` and `i8mm` matters in dollars-of-latency terms: the *same* int8 model is meaningfully faster on a core that has those instructions than on one that must widen int8 to int16 and lose half its lanes. Here is the idea in the kind of intrinsic a kernel actually uses:

```cpp
// One ARM NEON int8 dot-product: 16 int8 multiplies + adds into 4 int32 lanes.
#include <arm_neon.h>
int32x4_t mac_int8(int32x4_t acc, int8x16_t weights, int8x16_t acts) {
    // vdotq_s32 does, per output lane k: acc[k] += sum over j of w[4k+j]*a[4k+j]
    // i.e. 16 int8 MACs folded into ONE instruction, accumulating in int32.
    return vdotq_s32(acc, weights, acts);
}
// The fp32 equivalent (vmlaq_f32) processes only 4 lanes per instruction,
// so the int8 path does ~4x the MACs per issue slot on the SAME core.
```

**On the GPU (warps / SIMT).** A mobile GPU runs threads in lockstep groups (warps/wavefronts of 32 or 64). Each thread does scalar work, but the hardware coalesces memory accesses and packs ALU ops. Mobile GPUs are fp16-native (2× the fp32 rate via packed fp16), and some support int8 dot-product (`dp4a`-style) instructions. The win is again partly data width and partly that more values per fetched cache line means fewer fetches — the bandwidth argument from §7.

**On the NPU (systolic / MAC arrays).** This is the elegant one. A systolic array is a 2-D grid of tiny processing elements (PEs), each holding one weight, wired only to its neighbors. Activations *flow* across the grid; partial sums accumulate down it. The genius is **reuse**: a weight loaded into a PE stays put (weight-stationary dataflow) and is multiplied against the entire stream of activations that passes through that PE. So one expensive DRAM read of a weight is amortized across hundreds of cheap MACs — exactly the §7 prescription for beating the bandwidth wall — and the activations, once on-chip, ripple through the whole array without going back to memory. Google's TPU is the famous example (a 256×256 array); mobile NPUs use smaller arrays of the same idea. Figure 3 shows a tiny 2×2 version of the dataflow.

![A grid diagram of a small systolic array showing activations entering from the left rows, processing elements each holding a weight and performing a multiply-accumulate, partial sums flowing downward, and outputs leaving at the bottom](/imgs/blogs/the-edge-hardware-landscape-3.png)

For int8 specifically, the array packs more PEs in the same area (smaller multipliers) *and* moves half the bytes for weights and activations, so it sustains roughly 2× the fp16 rate — and because the dataflow already kills most DRAM traffic, that throughput is real, not theoretical-peak-only. The reason NPUs hate dynamic shapes now makes physical sense too: the *exact* mapping of which weight sits in which PE and when each activation arrives is planned ahead of time for a fixed shape. Change the shape and the schedule is wrong.

## 9. Tier 4 — single-board computers and edge GPUs: the watts-to-TOPS sweet spot

Above the phone but below the datacenter sits the most fun tier for builders: single-board computers and small edge GPUs you can hold in your hand and run off a barrel-jack power supply.

- **Raspberry Pi 5**: a quad-core Cortex-A76 at 2.4 GHz, 4–8 GB LPDDR4X (~17 GB/s bandwidth), no ML accelerator on board, ~5–8 W, about \$60–80. It is a *CPU* target — NEON int8 is your friend, and a quantized MobileNet runs in a few tens of ms. Pair it with a Coral USB accelerator and you add 4 TOPS int8 over USB.
- **Google Coral** (Edge TPU, dev board or USB stick): **4 TOPS int8** at ~2 W (≈2 TOPS/W), int8-only, ~\$60–80. Brilliant for a fixed CNN; useless for anything the Edge TPU compiler cannot map.
- **NVIDIA Jetson Orin Nano**: the entry CUDA edge GPU. The 8 GB module is rated around **40–67 TOPS int8** (sparse), 8 GB LPDDR5 at ~68–102 GB/s, configurable **7–15 W**, dev kit around \$250–500. You get the full CUDA / TensorRT stack, so this is where serving-grade optimization (TensorRT engines, fp16/int8, batching) meets the edge.
- **NVIDIA Jetson AGX Orin**: the flagship. Up to **275 TOPS int8** (sparse), up to 64 GB LPDDR5 at ~204 GB/s, configurable **15–60 W**, around \$2000. This runs real LLMs, multi-camera detection, and serious robotics perception on-device.

The reason this tier is the sweet spot is the **TOPS-per-watt** story together with a *real memory system*. A Jetson Orin Nano gives you tens of TOPS at a battery-feasible 15 W, with enough DRAM bandwidth (~68–102 GB/s) and capacity (8 GB) to actually hold and feed mid-sized models — and the full NVIDIA software stack (CUDA, cuDNN, TensorRT) you already know from the server. You do not have to fight a fragmented mobile NPU API; you build a TensorRT engine and run it. The cost is power: 15–60 W needs active or substantial passive cooling, which is fine on a drone or a robot or a kiosk but not on a coin cell. Figure 6 puts the tiers' compute, memory, and power side by side so the scaling is explicit.

![A matrix comparing microcontroller, mobile NPU, Jetson Orin, and FPGA or ASIC tiers across compute, memory, and power columns, showing compute rising from sub-gigaflop to hundreds of TOPS while power rises from milliwatts to tens of watts](/imgs/blogs/the-edge-hardware-landscape-6.png)

The honest mental model for this tier: a Jetson AGX Orin is a small, power-capped datacenter GPU with a unified-memory twist (CPU and GPU share the LPDDR, so there is no PCIe copy — a real advantage for memory-bound workloads). If you have shipped on server GPUs, your TensorRT and CUDA skills transfer almost directly; the new constraints are the power cap and the shared (not dedicated, not enormous) memory bandwidth.

The unified-memory point deserves one more beat because it changes how you size a workload. On a discrete server GPU, getting a tensor onto the device costs a PCIe copy, and total memory is split between host and device. On a Jetson, the CPU and GPU address the *same* LPDDR, so a producer-consumer pipeline (camera capture on CPU → preprocessing → GPU inference → CPU post-processing) can pass tensors by reference with zero-copy buffers, and the GPU can in principle use the *entire* module memory (up to 64 GB on the AGX). That is why a Jetson AGX Orin can hold a 13B-parameter LLM that would not fit on many discrete edge GPUs of similar compute: the bottleneck moves from VRAM capacity to the shared LPDDR bandwidth, which — per the §7 machine-balance table — is the thing that actually caps decode tokens/s anyway. So when you size a model for a Jetson, budget against *total module memory and bandwidth*, not a separate VRAM figure, and remember that the CPU side of your pipeline is competing for that same bandwidth in real time. The power knob (`nvpmodel`) interacts with all of this: dropping from MAXN to the 15 W cap lowers both the GPU clock and the memory clock, so a memory-bound model can lose throughput from the *bandwidth* reduction even when you thought you were only trading away compute.

## 10. Tier 5 — FPGAs and ASICs: when custom silicon earns its keep

At the top of the customization axis sit **FPGAs** (field-programmable gate arrays — reconfigurable logic you wire into a custom dataflow) and **ASICs** (application-specific integrated circuits — silicon etched for exactly one job). These are not "faster CPUs"; they are a different computing paradigm.

A conventional CPU/GPU is **von Neumann**: instructions and data live in memory, are fetched, decoded, and executed by general units, and intermediate results bounce back to memory between operations. Every op pays instruction-fetch and memory-round-trip overhead. A **dataflow** architecture — what you build on an FPGA or bake into an ASIC — instead lays out the *computation graph in space*: each operation is a physical block, data streams from block to block through on-chip wires and buffers, and there is no instruction stream and minimal DRAM traffic. For a fixed network this is enormously efficient: you spend transistors only on the ops you use, in exactly the dataflow that keeps data on-chip (the §7 dream). Microsoft's Project Brainwave (FPGA inference) and the entire wave of inference ASICs (Google's Edge TPU, Hailo, countless startups) are this idea.

When is custom silicon worth it? The trade-off is **flexibility and NRE cost versus efficiency**:

- **FPGA** makes sense when you have a fixed, high-value model, need deterministic ultra-low latency (no OS jitter), want a precision the commodity parts do not offer (binary/ternary nets, exotic bit-widths), and can absorb the hard engineering (HDL or HLS) of mapping a network to gates. Power and per-unit cost are moderate; the development cost is high.
- **ASIC** makes sense only at **volume**, because the non-recurring engineering (NRE) — design, masks, tape-out — runs into the millions of dollars. Amortized over tens of millions of units (a phone NPU, a Coral) it is the cheapest, most efficient option per inference; for a thousand units it is absurd. The mobile NPUs in §5 *are* ASICs that won the volume bet.

For the model optimizer, the takeaway is narrow but important: most of you will never tape out silicon, but you *will* target an ASIC (every phone NPU and Edge TPU is one), and the rules — fixed op set, fixed precision, static shapes, dataflow reuse — are exactly the §5–§8 rules, just frozen harder into the hardware. The more custom the silicon, the less the model is allowed to surprise it.

## 11. Practical: how to query what you are running on

Enough taxonomy. The skill that pays off is *reading these numbers off a live device* instead of trusting a spec sheet (or your memory). Here are the real commands and APIs, by tier.

**A CUDA edge GPU (Jetson) from Python.** PyTorch exposes the device properties directly:

```python
import torch

assert torch.cuda.is_available(), "no CUDA device visible"
props = torch.cuda.get_device_properties(0)
print(f"name:                {props.name}")
print(f"compute capability:  {props.major}.{props.minor}")
print(f"total memory:        {props.total_memory / 1e9:.2f} GB")
print(f"multiprocessors:     {props.multi_processor_count}")
# Memory bandwidth is not in props directly; derive it from the bus:
#   bandwidth (GB/s) ~= mem_clock_hz * bus_width_bits / 8 / 1e9 * 2  (DDR)
# On Jetson, read it from the system instead (see tegrastats below).
```

**Jetson live telemetry: `tegrastats`.** This is the single most useful command on a Jetson. It streams GPU/CPU/EMC (memory controller) utilization, clocks, power rails, and temperatures in real time:

```bash
# Stream every 1000 ms; watch the GR3D_FREQ (GPU) and EMC_FREQ (memory) lines.
sudo tegrastats --interval 1000

# Typical line (abbreviated):
#   RAM 3200/7860MB ... EMC_FREQ 0%@2133 GR3D_FREQ 12%@624 ... VDD_GPU_SOC 1450mW ...
# EMC_FREQ near 100% while GR3D_FREQ is low  ==>  you are MEMORY-BOUND.
# GR3D_FREQ near 100% while EMC_FREQ is low  ==>  you are COMPUTE-BOUND.
```

That EMC-versus-GR3D reading is the roofline made observable: if the memory controller is pinned and the GPU is idle, no amount of FLOP-cutting will help you. Use `sudo nvpmodel -q` to see and set the power mode (the 7 W / 15 W / MAXN caps), because the same Jetson is a different machine at 7 W than at 60 W.

**Mobile CPU: read `/proc/cpuinfo` for SIMD features.** On any ARM Linux/Android shell, the `Features` line tells you which SIMD paths your kernels can use:

```bash
# On the device (adb shell on Android, or directly on a Pi):
grep -m1 Features /proc/cpuinfo
# Look for:  asimd   -> NEON (Advanced SIMD) present
#            asimddp -> int8 dot-product (SDOT/UDOT) present  <-- the int8 fast path
#            asimdhp -> native fp16 arithmetic
#            i8mm    -> int8 matrix-multiply extension (newest, big int8 win)
lscpu | grep -i flags   # on a Pi 5 / aarch64 Linux, same idea
```

If `asimddp` is missing, your int8 kernels fall back to a slower widening path and the int8 advantage shrinks — a concrete example of why you query the *actual* core, not the marketing name.

**Android: enumerate NNAPI / TFLite delegates and what they can accelerate.** Before you assume the NPU will take your model, ask the runtime which ops it will delegate. With the TFLite interpreter you select a delegate and can inspect partitioning:

```python
import tensorflow as tf

# Try the GPU delegate; fall back to CPU XNNPACK if it is unavailable.
try:
    delegate = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
    interpreter = tf.lite.Interpreter(
        model_path="model_int8.tflite",
        experimental_delegates=[delegate],
    )
    backend = "GPU delegate"
except (ValueError, OSError):
    # XNNPACK (optimized CPU) is the default; NNAPI is legacy on new Android.
    interpreter = tf.lite.Interpreter(model_path="model_int8.tflite", num_threads=4)
    backend = "CPU / XNNPACK"

interpreter.allocate_tensors()
print(f"running on: {backend}")
# To see the partition plan, build with --define=tflite_with_xnnpack=true and
# read the verbose log: it prints "Replacing N node(s) with delegate ..." per
# partition. MANY partitions == many fallback handoffs (the §6 cliff).
```

On newer Android the modern path is a vendor delegate (Qualcomm QNN, MediaTek NeuroPilot) or LiteRT's accelerator service rather than the now-deprecated NNAPI, but the diagnostic is identical: *count the partitions*.

**Apple: pick the compute unit with `coremltools` (CPU / GPU / ANE).** Core ML lets you constrain which engine runs the model, which is exactly how you measure the per-tier gap and confirm the model maps to the Neural Engine:

```python
import coremltools as ct

# Convert with a chosen compute precision and unit.
mlmodel = ct.convert(
    traced_torch_model,
    inputs=[ct.TensorType(shape=(1, 3, 224, 224))],
    compute_precision=ct.precision.FLOAT16,   # ANE is fp16/int8
    # CPU_AND_NE forces "ANE if possible, CPU otherwise" -- great for measuring fit.
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
mlmodel.save("model_ane.mlpackage")

# To A/B the tiers, reload the SAME model pinned to each engine and time it:
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE):
    m = ct.models.MLModel("model_ane.mlpackage", compute_units=unit)
    # time m.predict(sample) over many runs after warm-up; compare p50.
```

If `CPU_AND_NE` is no faster than `CPU_ONLY`, your model is *not* actually running on the Neural Engine — Xcode's Core ML performance report (the per-layer "compute unit" column) tells you which layers fell back and why. That tool is the Apple analog of counting TFLite partitions.

The throughline: every tier exposes a way to ask "what am I, and where did my graph actually run?" Learn the one for your target. The most expensive edge bugs come from *assuming* the accelerator ran your model when half of it quietly fell back to the CPU.

## 12. A worked target-fit walkthrough: where does my model go?

Let us put the whole framework to work as a decision, the way you would in a planning meeting. Figure 8 is the decision tree; the prose below is the reasoning behind it.

![A decision tree for choosing an edge target that starts from whether the model footprint fits on-chip, branches on whether all operators are supported on the accelerator, and ends at microcontroller, phone NPU, or Jetson and laptop targets](/imgs/blogs/the-edge-hardware-landscape-8.png)

The tree's first question is deliberately **not** "how many FLOPs?" It is "does the working set fit on-chip, and is the op set accelerator-friendly?" — because, as §7 proved, footprint and op-fit decide the outcome far more often than arithmetic does. Walk three candidates through it.

#### Worked example: three models, three tiers

**(a) A 5 MB int8 keyword-spotter for an always-on doorbell.** Weights 5 MB, peak activation ~200 KB, all ops standard (conv, depthwise, fully-connected). Footprint check: 5 MB does not fit MCU SRAM but *does* fit external flash; the 200 KB activation arena fits a Cortex-M7's 1 MB SRAM. Op check: all CMSIS-NN-supported int8 ops. Power: always-on means **mW matters most**. Verdict: **MCU (Cortex-M7 + CMSIS-NN / TFLite-Micro)**. Even though a phone would run it in microseconds, the doorbell has no phone-class SoC and a tiny power budget. The MCU wins on \$ (≈\$2 chip) and on milliwatts. This is the TinyML sweet spot, and MLPerf Tiny is the benchmark that would score it.

**(b) A 12 MB int8 MobileNetV3 for real-time camera segmentation in a phone app.** Footprint: fits the phone's GB of LPDDR trivially. Op check: MobileNet is the *canonical* NPU-friendly graph — static shapes, standard ops — *except* if your head has an exotic op. Power: a phone, so per-inference energy and thermals matter. Verdict: **phone NPU via the int8 delegate**, falling back to the **GPU** (not CPU) for any unsupported tail op. Expected payoff (from §3–§5 and figure 4): ~3 ms p50 on the NPU versus ~25 ms on the CPU, at roughly 0.3 W versus 2.5 W. If a single op forces a CPU partition (the §6 cliff), you fix the op before you ship.

**(c) A 7B-parameter chat assistant quantized to 4-bit (Q4_K_M GGUF), ~4 GB weights.** Footprint: 4 GB does not fit any MCU and is tight even on an 8 GB phone once you add the OS and KV cache; it is comfortable on a laptop or a Jetson Orin (8–64 GB). Op check: it is a transformer decoder — fine for GPUs, awkward for fixed NPUs because of dynamic sequence length (the §5 dynamic-shape problem). Bound check (from the §7 worked example): **memory-bound** decode, so we care about bandwidth, not TOPS. Verdict: **Jetson Orin / a laptop GPU running `llama.cpp` (GGUF) or MLC-LLM**, picking the lowest-bit quantization that holds accuracy because that is what raises decode tokens/s. Not a phone (too tight on memory and bandwidth), definitely not an MCU.

```bash
# Candidate (c): build and run a 4-bit GGUF LLM on a Jetson / laptop with llama.cpp.
# Quantize an fp16 GGUF to Q4_K_M (4-bit k-quant, the on-device default):
./llama-quantize ggml-model-f16.gguf ggml-model-Q4_K_M.gguf Q4_K_M

# Run with all layers offloaded to the GPU (-ngl 999) and time tokens/s:
./llama-cli -m ggml-model-Q4_K_M.gguf -ngl 999 -p "Summarize edge AI." -n 128
# Watch tokens/s in the timing footer; on a memory-bound decode it tracks
# (effective bandwidth) / (bytes per weight read per token), exactly as §7 predicts.
```

Notice what drove every decision: footprint, op-fit, bound (memory vs compute), and power — never the raw FLOP count alone. That is the discipline this whole post is trying to install. For *how* to actually shrink model (c) to fit and run faster, the quantization track ([a-taxonomy-of-model-compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) and the per-technique posts) is the next stop; this post just tells you which target you are shrinking *for*.

## 13. Results and comparison tables

Two tables. The first is the reference card — the four numbers per tier — so you can size a target at a glance. The second is the punchline: the *same* model across CPU, GPU, and NPU on one phone, which is the measurement that proves the per-tier gap is real and not a spec-sheet fantasy.

### The tier reference card

| Tier | Peak compute | Memory | Bandwidth | Power | Price | Typical model |
| --- | --- | --- | --- | --- | --- | --- |
| Cortex-M0+/M4 MCU | <0.1 GMAC/s, often no FPU | 16–256 KB SRAM | on-chip only | mW | ~\$1 | keyword spotting, anomaly detection |
| Cortex-M7 MCU | ~0.1–0.5 GMAC/s int8 | ~0.5–1 MB SRAM, 1–2 MB flash | on-chip + slow QSPI | tens of mW | ~\$2–10 | MCUNet-class CNN, MLPerf Tiny |
| Mobile CPU (big.LITTLE) | low tens of GOP/s int8 (NEON) | shared GB LPDDR | ~40–60 GB/s shared | 2–5 W | (in SoC) | universal fallback, glue ops |
| Mobile GPU (Adreno/Mali) | 1–3 TFLOP/s fp16 | shared GB LPDDR | ~40–60 GB/s shared | 2–4 W | (in SoC) | flexible NN, fp16, dynamic shapes |
| Mobile NPU (ANE/Hexagon/APU) | ~10–40 TOPS int8 | shared LPDDR | ~40–60 GB/s shared | 0.3–2 W | (in SoC) | static int8 CNN/transformer |
| Coral Edge TPU | 4 TOPS int8 | host RAM | over USB/PCIe | ~2 W | ~\$60–80 | fixed int8 CNN |
| Jetson Orin Nano | ~40–67 TOPS int8 (sparse) | 8 GB LPDDR5 | ~68–102 GB/s | 7–15 W | ~\$250–500 | detection, mid LLM, robotics |
| Jetson AGX Orin | up to 275 TOPS int8 (sparse) | up to 64 GB LPDDR5 | ~204 GB/s | 15–60 W | ~\$2000 | multi-cam, 7B–13B LLM on-device |
| FPGA / inference ASIC | tuned (binary→int8 dataflow) | on-chip BRAM + DRAM | architecture-dependent | 1–25 W | NRE + per-unit | fixed, ultra-low-latency, custom precision |

Every "TOPS" above is int8 peak, and the Jetson sparse figures assume 2:4 structured sparsity; dense rates are roughly half. Shared-bandwidth tiers (phone) all draw from the *same* LPDDR pool, which is why three tiers list the same ~40–60 GB/s — a structural fact worth remembering when you stack workloads.

### Same model, three engines, one phone

These are *representative* numbers for a MobileNet-class classifier (≈3–6M params) at int8, batch 1, after warm-up, on a recent flagship Android phone, in the spirit of public TFLite / MLPerf Mobile benchmarks. **Treat them as approximate, order-of-magnitude figures** to show the *shape* of the gap, not a spec for any one device:

| Engine | Precision | p50 latency | Rel. speed | Active power | Energy / inference |
| --- | --- | --- | --- | --- | --- |
| CPU (1 big core, XNNPACK) | int8 | ~18–30 ms | 1× | ~2.5 W | ~50–75 mJ |
| CPU (4 threads) | int8 | ~8–14 ms | ~2–3× | ~3.5 W | ~35–50 mJ |
| GPU (delegate) | fp16 | ~6–10 ms | ~3× | ~3 W | ~20–30 mJ |
| NPU (delegate) | int8 | ~2–4 ms | ~6–10× | ~0.3–0.6 W | ~1–2 mJ |

Two facts jump out. The NPU is not just *faster*; it is **10–50× more energy-efficient per inference** than the CPU, which on an always-on or battery-bound product is the number that decides the product, not the millisecond count. And the GPU sits sensibly in between — your gentle fallback. How would you *measure* this honestly? Pin the workload to one engine at a time (the §11 Core ML / delegate selection), discard the first dozen inferences (warm-up: kernels compile, clocks spin up, caches fill), run a few hundred and report **p50 and p99** (the tail is where thermal throttling and scheduler migrations show up), and watch the temperature — on a phone, a model that runs at 3 ms cold can run at 8 ms after 30 seconds of sustained load as the chip throttles. A single cold "best ever" number is the most common way edge benchmarks lie.

## 14. Case studies: real numbers from shipped work and the literature

Four results that anchor the tiers in reality. I have kept the headline numbers to ones reported in the source papers/products; where I round, I say approximate.

**MCUNet on a Cortex-M7 (Lin et al., NeurIPS 2020).** Co-designing TinyNAS + TinyEngine, the authors ran ImageNet classification on an STM32F746 (Cortex-M7, 320 KB SRAM, 1 MB flash) at roughly **70.7% top-1**, fitting the activation peak inside the SRAM budget that prior models blew past. The lesson is the §2 one made concrete: the binding constraint was *peak SRAM of the activation graph*, and the win came from co-designing the model with the memory-aware runtime, not from FLOP reduction alone.

**MobileNetV3 on a Pixel phone (Howard et al., 2019).** MobileNetV3-Large reached ~75% ImageNet top-1 at roughly **~50–60 ms on a Pixel CPU**, with the "Small" variant much faster — explicitly tuned via platform-aware NAS *and* hardware-friendly ops (hard-swish chosen partly because it is cheaper to implement on-device than swish). It is the textbook case of letting the target's op costs shape the architecture, the bridge between this post and the efficient-architecture lever.

**Coral Edge TPU on MobileNetV2.** Google's published Coral benchmarks put MobileNetV2 at roughly **~2.5–3 ms** per inference on the Edge TPU at ~2 W — versus tens of ms on an embedded CPU — but *only* for the int8-compiled, fully-mapped model. Any op the Edge TPU compiler cannot place runs on the host CPU and the latency reverts to CPU-class. It is the §6 fallback cliff as a product constraint: the 3 ms exists only if 100% of the graph maps.

**Whisper.cpp / llama.cpp on a laptop (community + Gerganov).** A 7B LLaMA-class model at 4-bit (Q4_K_M) runs at roughly **15–40 tokens/s on an Apple-silicon laptop**, and Whisper "base/small" transcribes faster than real time on a CPU — both because the 4-bit quantization cut the *bytes per weight* and the decode is memory-bound (§7's worked example). These are the clearest proof that on memory-bound edge inference you optimize bandwidth/bytes, and that a laptop or Jetson — not a phone, not an MCU — is the right tier for a multi-GB LLM.

## 15. When to reach for which target (and when not to)

A decisive recommendation section, because "it depends" helps no one. Pick the *lowest* tier that meets the constraint, because lower tiers are cheaper and more power-efficient — but never below the tier where your model's footprint or op set stops fitting.

- **Reach for an MCU** when power is the dominant constraint (always-on, battery/coin-cell, mW budget), the model is small and int8, and the ops are standard CNN/DSP. **Do not** reach for an MCU for anything over a few MB of weights with fat activations, for transformers/LLMs, or when you need fp16 — you will spend the project fighting SRAM.
- **Reach for the mobile CPU** as the universal fallback, for dynamic-shape models, and for the glue/pre-/post-processing. **Do not** make it your main inference engine if an NPU or GPU is available — it is the slowest and most power-hungry path, and it will thermal-throttle.
- **Reach for the mobile GPU** when you need fp16 fidelity, portability across many phones, dynamic shapes, or a *graceful* fallback from the NPU. **Do not** assume it beats the NPU on a clean int8 CNN — it usually does not, on either latency or energy.
- **Reach for the mobile NPU** for a static, int8-friendly graph of standard ops — it is the latency and (especially) energy winner. **Do not** target it for dynamic shapes, exotic ops, or fp32-sensitive models without first counting partitions; one fallback op (§6) can erase the entire win.
- **Reach for a Jetson / edge GPU** when you have watts to spend (a robot, drone, kiosk, camera with mains/large battery), need a real memory system and CUDA/TensorRT, or must run an LLM or multi-model pipeline on-device. **Do not** reach for it on a coin-cell product or where \$250–2000 and 15–60 W are unacceptable.
- **Reach for an FPGA** for a fixed, high-value model needing deterministic ultra-low latency or a custom precision the commodity parts lack — if you can absorb the HDL/HLS engineering. **Reach for an ASIC only at volume** (millions of units) where NRE amortizes; otherwise you are buying a phone NPU or Edge TPU, which is someone else's ASIC bet you get to ride for free.

The capstone post, [the-edge-optimization-playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook), turns this target choice plus the four levers into an end-to-end recipe; this post is its first step — *know the silicon* — and everything else (which lever, how aggressively to quantize, whether to prune) is a function of the answer.

## 16. Key takeaways

- **Three numbers define a target**: peak compute, memory capacity, and memory bandwidth — plus a power envelope. Memorize all three for your device before you touch the model.
- **Data movement, not arithmetic, dominates energy.** A DRAM access (~640 pJ) costs ~3000× an int8 MAC (~0.2 pJ). Every accelerator is a machine for avoiding DRAM, and your optimizations should be too.
- **Compute-bound vs memory-bound is set by arithmetic intensity vs machine balance** ($I$ vs $P/B$). Below the balance, cutting FLOPs does nothing — cut *bytes* (quantize) instead. This is why FLOP-counting optimizations backfire.
- **NPUs are int8-first, static-shape, limited-op accelerators.** They reward a clean int8 graph with ~10× latency and ~10–50× energy wins over the CPU, and punish unsupported ops with a CPU-fallback cliff that can erase it.
- **int8 is 2–4× fp16 on the same die** because of more lanes per register, dedicated int8 MAC paths, and half the memory traffic — a hardware fact you should design models to exploit.
- **Pick the lowest tier that fits** (footprint + op set), then optimize for that tier's scarce resource — mW on an MCU, bandwidth on a phone NPU, watts on a Jetson.
- **Always query the live device** (`tegrastats`, `/proc/cpuinfo` features, delegate partition counts, Core ML compute-unit reports) and confirm where your graph *actually* ran. Assuming the accelerator ran it is the most expensive edge bug.
- **Measure honestly**: warm up, report p50 and p99, watch thermal throttling, and account for shared bandwidth. A single cold best-case number is how edge benchmarks lie.

## 17. Further reading

- **Mark Horowitz, "Computing's Energy Problem (and what we can do about it)," ISSCC 2014** — the source of the energy-per-operation numbers; the foundation for *why* data movement dominates.
- **Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," CACM 2009** — the roofline model formalized; pairs with the within-series roofline post.
- **Lin et al., "MCUNet: Tiny Deep Learning on IoT Devices," NeurIPS 2020** — the TinyML co-design landmark; the SRAM-as-binding-constraint lesson.
- **Howard et al., "Searching for MobileNetV3," ICCV 2019** — platform-aware architecture, hardware-friendly ops, on-device latency targets.
- **Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017** — the systolic-array accelerator explained by its designers; the canonical MAC-array reference.
- **MLPerf Tiny and MLPerf Mobile benchmark suites (MLCommons)** — the standardized, honest measurement methodology for MCU- and phone-class inference.
- **NVIDIA Jetson documentation and the TensorRT developer guide** — the official numbers and the `tegrastats` / `nvpmodel` tooling for the edge-GPU tier.
- **Apple Core ML and Core ML Tools documentation** — compute-unit selection, the Neural Engine, and the Xcode performance report for confirming on-device placement.
- Within this series: [a-taxonomy-of-model-compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) (the four levers and the Pareto frame), [the-roofline-model-where-your-bottleneck-lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) (the bound analysis in full), and the capstone [the-edge-optimization-playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
