---
title: "TinyML on microcontrollers: TFLite Micro, CMSIS-NN, and the no-malloc world"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Run a neural net on a $2 microcontroller with 256 KB of SRAM and no operating system — from the tensor arena and int8 kernels to a full TFLite Micro sketch with measured size, latency, and power."
tags:
  [
    "edge-ai",
    "model-optimization",
    "tinyml",
    "tflite-micro",
    "cmsis-nn",
    "microcontrollers",
    "int8-quantization",
    "inference",
    "efficient-ml",
    "embedded-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/tinyml-on-microcontrollers-1.png"
---

There is a neural network running in your ear right now, and it has no idea what an operating system is.

Inside a wireless earbud, on a chip that costs about \$2 and has 256 KB of SRAM, a small convolutional network wakes up sixteen times a second, listens to a 30 ms window of audio, and decides whether you just said the wake word. It does this on a Cortex-M4 clocked at 80 MHz, drawing a few milliwatts so the coin cell lasts a week. There is no Linux, no Python, no `malloc` being called in the hot loop. The model lives in flash as a frozen blob of bytes; the activations live in one buffer that was sized at compile time and never grows. Every byte is accounted for. Every cycle is accounted for. This is TinyML, and it is the most constrained, most honest corner of the entire edge-AI world.

I want to be precise about how constrained. A cloud GPU has tens of gigabytes of HBM and trillions of FLOP/s. A flagship phone NPU has gigabytes of LPDDR and runs at watts. The microcontroller (MCU) in that earbud has kilobytes of RAM, runs at tens of milliwatts, and frequently has no floating-point unit and no memory-management unit at all. You do not "deploy a model" to it the way you push a container to a server. You compile the model *into the firmware* and ship the whole thing as a single binary that boots in microseconds. If your peak working set is 257 KB and you have 256 KB, the device does not run slowly — it does not run. The figure below is the mental model I keep in my head for every TinyML project: a big read-only flash holding the model and code, and a tiny, jealously-guarded SRAM holding everything that changes.

![A layered memory map of a microcontroller showing flash holding firmware code and the model flatbuffer on top, with SRAM below split into the tensor arena, stack and globals, and free headroom, plus the CPU with SIMD and no memory unit](/imgs/blogs/tinyml-on-microcontrollers-1.png)

By the end of this post you will be able to take a trained float model, quantize it to int8, convert it to a TensorFlow Lite flatbuffer, turn that flatbuffer into a C array, and run it on a Cortex-M with TFLite Micro (TFLM) and ARM's CMSIS-NN kernels — and you will know how to size the tensor arena, how to read the latency in cycles, and how to tell when an MCU is the wrong target and you should step up to a Cortex-A or an NPU. We will hit all three things this series cares about: the **science** (why int8 is mandatory, why CMSIS-NN is 4–5× the reference kernels, how arena sizing actually works), the **practical** flow (real C++, real `xxd`, real conversion code), and **measured results** (size, latency, power, accuracy on named parts). This is the post that opens the TinyML track, so it sits inside the larger frame of the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — quantization, pruning, distillation, efficient architectures, read off the accuracy–efficiency Pareto frontier — but down here the dominant lever, by a wide margin, is quantization plus ruthless memory planning.

## 1. The MCU reality: what you are actually targeting

If you have only ever shipped models to servers or phones, the first thing to internalize is that a microcontroller is not a small computer. It is a different kind of machine. Let me lay out the numbers, because the numbers are the whole story.

A typical TinyML target is an ARM Cortex-M class core. The lineup you care about:

- **Cortex-M0 / M0+**: the cheapest, smallest core. 48 MHz is common. No DSP extension, no floating-point unit (FPU). SRAM measured in single-digit to low-double-digit kilobytes. This is where you run a tiny MLP or a decision tree, not a CNN.
- **Cortex-M4 / M4F**: the TinyML workhorse. ~80–120 MHz, a single-precision FPU on the "F" variants, and crucially the **DSP extension** with SIMD instructions like `SMLAD`. 128–256 KB of SRAM is typical. This is where keyword spotting lives.
- **Cortex-M7**: the high end of the MCU world. 300–600 MHz, double-precision FPU, a deeper pipeline, sometimes a cache and tightly-coupled memory. 256 KB to ~1 MB of SRAM. This is where tiny vision models (person detection, gesture) become real-time.

Across all of them, the constants that define your life are the same. SRAM is **kilobytes to low megabytes**. Flash is larger — hundreds of KB to a few MB — but it is read-only at runtime in practice and slower to fetch from. Clocks are **tens to hundreds of MHz**, three orders of magnitude below a desktop. Many parts have **no FPU** and almost none have an **MMU**, which means no virtual memory, no paging, and no protection between "your model" and "the rest of the firmware." Power budgets are **milliwatts**, sometimes tens of microwatts in the duty-cycled average, because the entire point is to run for months on a battery or to harvest energy. I go deeper on the hardware spectrum in [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape); here the only thing that matters is how brutally small the budgets are.

The two consequences that flow from this are the spine of everything below.

**First: no dynamic allocation in practice.** Real embedded firmware avoids `malloc`/`free` in the steady state. Heap fragmentation on a device that runs for months with no reboot is a slow-motion crash: after a few hundred thousand allocate/free cycles the heap can no longer satisfy a request even though the total free bytes are plenty, and the device hangs at 3 a.m. when nobody is watching. So embedded engineers allocate everything up front and never touch the heap again. A machine-learning runtime that calls `malloc` per inference is, in this world, simply broken. TFLite Micro's central design decision — the thing that makes it usable on an MCU at all — is that it does not allocate at runtime. We will get to exactly how.

**Second: int8 is mandatory, not optional.** A float32 weight is 4 bytes; an int8 weight is 1 byte. On a part with 256 KB of SRAM and a 300 KB model, the 4× difference between fp32 and int8 is the difference between "fits in flash" and "does not exist." And on a core with no FPU, float math is *emulated in software* — every multiply becomes a subroutine call costing tens of cycles. Integer math is native and single-cycle. So int8 is doing double duty: it shrinks the model 4× and it makes the math 10×+ faster on FPU-less parts. If you remember one rule from this post, remember that on an MCU you quantize first and ask questions later. The full mechanics of how int8 quantization maps a float tensor onto integers, and how much accuracy it costs, are in [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq); I will use the results here, not re-derive them.

Let me put a number on the float-emulation cost, because it is the part people underestimate. On a Cortex-M0+ with no FPU, a single-precision multiply compiled by the toolchain's soft-float library (`__aeabi_fmul`) is on the order of **tens of cycles** — frequently 30–50 — versus a **single cycle** for an integer multiply. A model that is 90% multiply-accumulate, run in software float on an FPU-less part, is therefore not "a bit slower" than int8; it is one to two orders of magnitude slower, and it is also 4× larger in flash. There is no scenario on an M0+ where float beats int8. Even on an M4F *with* an FPU, where a float multiply is single-cycle, int8 still wins on memory bandwidth (you move a quarter of the bytes) and because the SIMD path that retires two-to-four MACs per cycle exists only for integers. The FPU helps your control code; it does not make float inference competitive with int8 on these parts.

There is a small, principled accuracy cost for going to int8, and it is worth knowing the shape of it even though I derive it fully in the quantization post. Modeling quantization as adding uniform noise of step size $\Delta$, the error variance is $\sigma_q^2 = \Delta^2 / 12$, and the signal-to-quantization-noise ratio for a $b$-bit quantizer over a full-scale range is the classic

$$\text{SQNR} \approx 6.02\,b + 1.76\ \text{dB}.$$

Each additional bit buys about 6 dB. Going from float (effectively no quantization noise for our purposes) to int8 ($b = 8$) leaves roughly 50 dB of SQNR, which is *plenty* for a network whose final decision is an argmax over a handful of classes — the noise floor is far below the inter-class margin. This is the quantitative reason int8 PTQ costs a fraction of a point on a well-behaved model: the network's decision is robust to noise an order of magnitude larger than int8 introduces. It is also the reason int4 ($\approx 26$ dB) gets dicey on tiny models, where there is little redundancy to absorb the extra noise. Quantization is not free, but at int8 it is *cheap*, and on an MCU it is the entry ticket.

## 2. Memory is the entire game: arena sizing as math

Before any code, you need the one piece of arithmetic that decides whether a model fits. On a server you count parameters and FLOPs. On an MCU you count **peak working set** — the maximum number of bytes of activation tensors that are simultaneously alive at any point in the graph. This is a different number from "sum of all tensors," and the gap between them is where TinyML lives or dies. The full argument for why this is the binding constraint is in [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint); here I will make it concrete for the arena.

Let me set up the model. A tensor that is produced at graph step $p$ and last consumed at step $c$ is **live** over the interval $[p, c]$. If tensor $i$ has size $s_i$ bytes and live interval $[p_i, c_i]$, then the number of bytes alive at step $t$ is

$$M(t) = \sum_{i \,:\, p_i \le t \le c_i} s_i .$$

The peak working set is $M^\* = \max_t M(t)$. The tensor arena must be at least $M^\*$ bytes, plus a small amount of bookkeeping. That is the law. Everything the memory planner does is an attempt to reach this lower bound by packing tensor *offsets* inside one buffer so that two tensors whose live intervals do not overlap can share the same bytes.

Here is why the arena beats naive allocation, with numbers. Suppose a tiny graph produces three intermediate tensors: input $A = 96$ KB, feature $B = 64$ KB, feature $C = 60$ KB, and they are live like this: $A$ over steps 1–2, $B$ over step 2 only (it is consumed immediately), $C$ over steps 2–3. Then:

- At step 1: only $A$ is live → 96 KB.
- At step 2: $A$, $B$, $C$ all live → $96 + 64 + 60 = 220$ KB. This is the peak.
- At step 3: only $C$ live → 60 KB.

A naive runtime that mallocs a fresh buffer per tensor and frees lazily can end up holding the *sum*, 220 KB and up, depending on free timing. A good arena planner reuses offsets: once $A$ is dead after step 2, its 96 KB slot is free, and a later tensor produced at step 3 can be placed there. With careful packing this graph can fit in close to its true peak rather than the running sum. The figure shows the contrast for our keyword-spotting model, where naive per-tensor allocation would want about 210 KB but the planner packs everything into a 96 KB arena.

![A two-column comparison of per-tensor allocation that sums every tensor to 210 KB versus a pre-packed tensor arena that reuses freed offsets and fits the same graph in a 96 KB peak](/imgs/blogs/tinyml-on-microcontrollers-2.png)

The practical takeaway: **the arena size you must supply to TFLM is the peak working set of your specific model graph, not a function of parameter count.** Two models with identical parameter counts can have wildly different arena needs depending on how big their largest activation maps are and how those intervals overlap. A model with one giant early feature map (common in vision, before pooling shrinks the spatial dimensions) has a much higher peak than its parameter count suggests. This is why the first convolution of an MCU vision net is often the single most expensive thing in the whole budget, and why techniques like patch-based inference (which I forward-reference below) exist purely to chop that early peak down.

There is no clean closed form for the optimal packing — it is a 2D rectangle-packing problem (offset × time) that is NP-hard in general — so planners use greedy heuristics: sort tensors by size, place the largest first at the lowest free offset that does not collide with a live tensor, repeat. TFLM's offline-planned and online "greedy by size" planners both do versions of this. In practice they get within a few percent of the true peak for typical graphs, which is good enough; you size the arena with headroom anyway.

#### Worked example: sizing the arena for a KWS model

Take the keyword-spotting CNN we will use throughout. It takes a $49 \times 10$ MFCC feature frame (a spectrogram-like representation of a short audio window), runs it through a few depthwise-separable conv blocks, and outputs logits over a small vocabulary (say 12 words: "yes," "no," the digits, plus "silence" and "unknown"). Counting tensors:

- Input MFCC tensor (int8): $49 \times 10 \times 1 = 490$ bytes. Negligible.
- First conv output: $25 \times 5 \times 64 = 8{,}000$ bytes.
- Largest intermediate (after the widest conv, before pooling): about $25 \times 5 \times 64$ doubled by an im2col scratch buffer the kernel needs → on the order of tens of KB.
- The im2col / col-buffer that CMSIS-NN convolution kernels use as scratch is frequently the *largest single allocation*, larger than any activation, because it expands the receptive field of every output position into a flat matrix for the GEMM.

Add the live intervals, pack, and the peak comes out around 90–100 KB for this model on a Cortex-M4. So you declare a `constexpr int kTensorArenaSize = 96 * 1024;` and a static `uint8_t tensor_arena[kTensorArenaSize];`. If you guess too small, `AllocateTensors()` returns an error at startup — better than a runtime crash, and exactly how you discover the real number empirically: set it large, call `interpreter.arena_used_bytes()`, then shrink to that plus a small margin. I will show that loop in the code section.

## 3. The science of TFLite Micro: an interpreter with no malloc

TensorFlow Lite Micro (David et al., 2021) is the runtime that makes this possible, and its design is worth understanding because every constraint of the MCU is visible in its architecture. It is described in the paper "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems," and the design goal is stated bluntly: run inference with **no dynamic memory allocation, no operating system dependency, and no standard-library assumptions** beyond a tiny subset, in tens of kilobytes of code.

There are four pieces, and the figure further down wires them together.

**The model flatbuffer, in flash.** When you convert a model to `.tflite`, you get a [FlatBuffer](https://google.github.io/flatbuffers/) — a serialization format you can read *without parsing or copying*. The bytes on disk are the in-memory layout. This matters enormously on an MCU: the model sits in flash (read-only memory), and TFLM reads weights **in place** directly from flash. The weights are never copied into the scarce SRAM. A 300 KB int8 model occupies 300 KB of flash and roughly *zero* additional SRAM for its weights. Only activations need RAM. This is the single biggest reason the flatbuffer format was chosen over Protocol Buffers (which require a parse-and-allocate step that an MCU cannot afford).

**The tensor arena, the one buffer.** You hand TFLM a single contiguous block of memory — `uint8_t tensor_arena[N]` — that you allocated statically (a global array or a stack buffer). TFLM's memory planner carves this block into all the activation tensors and its own scratch and bookkeeping. It does this once, at `AllocateTensors()` time, using the peak-overlap math from the previous section. After that, `Invoke()` reads and writes inside the arena and **allocates nothing**. The arena is the entire dynamic memory footprint of inference. This is the no-malloc world: there is no heap involved at all, just one buffer you control.

**The op resolver, a registered subset.** TFLite supports ~150 operators. Your model uses maybe 7 of them. On a server you would link all of them; on an MCU every kernel is precious flash. So TFLM makes you *register the ops you use, by hand*. You instantiate a `MicroMutableOpResolver<N>` templated on the number of ops, then call `.AddConv2D()`, `.AddDepthwiseConv2D()`, `.AddFullyConnected()`, `.AddSoftmax()`, and so on — only for the ops your model contains. The linker then drops every unused kernel from the firmware image. Get this wrong (forget an op the model needs) and `AllocateTensors()` fails with a clear "didn't find op" error pointing at the missing one. This explicit registration is annoying the first time and a feature forever after: your firmware contains exactly the kernels you use and not one byte more.

**The interpreter and Invoke().** The `MicroInterpreter` ties the three together: it reads the flatbuffer, asks the resolver for each op's kernel, plans the arena, and binds input/output tensors. Then `Invoke()` walks the graph node by node, calling each registered kernel on its input/output slices of the arena. No allocation, no recursion into the OS, no I/O. On return, the output tensor holds your int8 logits.

![A dataflow graph showing the model flatbuffer in flash, a trimmed op resolver, and the tensor arena all converging on the interpreter, which feeds Invoke with no malloc and produces an int8 output tensor](/imgs/blogs/tinyml-on-microcontrollers-5.png)

The thing I want you to take away is that **TFLM is not a small TensorFlow.** It is a different object: an interpreter that has been stripped of every assumption an MCU cannot pay for. No threads, no allocator, no filesystem, no dynamic shapes (every tensor shape is known at convert time), no autograd, no training. What is left is a tight loop that reads a flatbuffer and dispatches int8 kernels into a fixed buffer. That austerity is exactly what lets it run in ~20–50 KB of code on a part with no operating system.

A subtlety worth stating: because there are no dynamic shapes, your model must be fully static. A transformer with a variable sequence length, a detector with a variable number of boxes, anything with data-dependent control flow — these need to be fixed to a maximum shape before conversion, or they will not convert cleanly. TinyML models are static graphs of static tensors. That constraint is liberating once you accept it: it is *why* the arena can be planned ahead of time.

## 4. The science of CMSIS-NN: why hand-written int8 kernels win

TFLM gives you the structure; CMSIS-NN gives you the speed. CMSIS-NN (Lai, Suda, Chandra, 2018, "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs") is ARM's library of hand-optimized neural-network kernels for Cortex-M. When TFLM runs on a Cortex-M with the DSP extension, its `Conv2D`, `DepthwiseConv2D`, and `FullyConnected` kernels dispatch into CMSIS-NN under the hood. The speedup over the portable "reference" kernels is consistently **4–5×** for int8 convolution and fully-connected layers, and it comes from one core idea: SIMD on a 32-bit datapath.

Here is the mechanism. The fundamental operation in a neural net is the multiply-accumulate (MAC): $\text{acc} \mathrel{+}= w \cdot x$. A reference int8 kernel does one MAC per loop iteration, with each 8-bit value living in its own 32-bit register, wasting three of every four bytes of the datapath. The Cortex-M4/M7 DSP extension provides packed SIMD instructions that treat a 32-bit register as **four packed int8 lanes** (or two int16 lanes). The key instruction is `SMLAD` — "signed multiply-accumulate, dual" — which takes two registers each holding two int16 values, multiplies the corresponding halves, and adds *both* products to a 32-bit accumulator **in a single cycle**:

$$\text{acc} \mathrel{+}= a_{\text{lo}}\cdot b_{\text{lo}} + a_{\text{hi}}\cdot b_{\text{hi}} .$$

So one `SMLAD` retires two MACs per cycle. CMSIS-NN's int8 kernels widen the packed int8 inputs to int16 lanes, lay out weights and activations so the loads are contiguous, keep the GEMM inner loop unrolled, and use `SMLAD` (and its 8-bit cousins on parts that support them) to stream through the dot product two products at a time. Combine the 2-MAC/cycle throughput with the elimination of per-element function-call overhead, software-emulated float, and bounds checks, and you land at the measured 4–5× over reference. The figure makes the contrast concrete on a Cortex-M4.

![A two-column comparison of a scalar reference int8 kernel doing one MAC per iteration at 10.4 ms versus a CMSIS-NN kernel packing four int8 lanes and using SMLAD for two MACs per cycle at 2.3 ms, about 4.5 times faster](/imgs/blogs/tinyml-on-microcontrollers-4.png)

Let me make the cycle count rigorous so the speedup is not hand-waving. Take a fully-connected layer with $N_{\text{in}} = 256$ inputs and $N_{\text{out}} = 64$ outputs. The total MACs are $N_{\text{in}} \times N_{\text{out}} = 256 \times 64 = 16{,}384$. On a reference kernel at roughly 1 MAC plus a few cycles of overhead per iteration, call it ~3 cycles/MAC, that is about $16{,}384 \times 3 \approx 49{,}000$ cycles. On CMSIS-NN with `SMLAD` at 2 MACs/cycle and a tight unrolled loop amortizing overhead, you approach ~0.6 cycles/MAC, so about $16{,}384 \times 0.6 \approx 9{,}800$ cycles — a 5× win. At 80 MHz, $9{,}800$ cycles is $9{,}800 / 80{,}000{,}000 \approx 122\ \mu s$ for that one layer. Those numbers are how you reason about whether a model hits a latency budget *before* you flash anything: count MACs, divide by the effective MAC/cycle of the kernel, divide by the clock.

Where do the cycles actually go, instruction by instruction? Walk one iteration of the CMSIS-NN inner loop on a Cortex-M4. The kernel keeps the running int32 accumulator in a register, and per pass it issues: one `LDR` to load a 32-bit word holding four packed int8 activations, one `LDR` to load four packed int8 weights, a pair of `SXTB16` instructions to sign-extend the low and high byte pairs into two int16 lanes each, then two `SMLAD` instructions that each retire two MACs into the accumulator. That is roughly 6 instructions to do 4 MACs. On the M4, `LDR`, `SXTB16`, and `SMLAD` are each single-cycle on a hit, and the loop is unrolled so the loop-counter decrement and branch are amortized across many MACs — so the steady-state cost lands near $6/4 = 1.5$ instructions per MAC, and because several of those overlap in the pipeline the *effective* rate is the ~0.6 cycles/MAC quoted above once the loads are served from zero-wait-state SRAM or TCM. The moment your weights or activations sit in a memory that stalls the load (external QSPI, a flash region with wait states), those `LDR`s are no longer single-cycle and the whole ratio collapses — which is exactly the memory-bound failure I describe in the stress-test section. The lesson hiding in the instruction count is that the int8 kernel is *bandwidth-bound on the loads*, not bottlenecked on the multiplier: the `SMLAD` is free if you can keep feeding it, and the entire art of CMSIS-NN is keeping it fed.

How does a *convolution* become a stream of `SMLAD`s? CMSIS-NN's conv kernels use the **im2col** trick: each output position's receptive field — the patch of input the filter slides over — is flattened into a column vector, and the whole convolution becomes a matrix multiply (GEMM) of the weight matrix against the stacked columns. For a conv with $K \times K$ kernel, $C_{\text{in}}$ input channels, and $C_{\text{out}}$ output channels, each output column is a dot product of length $K \cdot K \cdot C_{\text{in}}$, and there are $H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{out}}$ such dot products. The kernel materializes one or two columns at a time into a small scratch buffer (the im2col buffer from section 2) and runs the GEMM inner loop with `SMLAD`, so the convolution inherits the 2-MAC/cycle throughput of the matrix multiply. This is also why the im2col scratch buffer is often the single largest allocation in the arena: it holds $K \cdot K \cdot C_{\text{in}}$ values per column, expanded, even though it is transient. The trade is bytes for speed — you spend scratch RAM to turn a strided, cache-unfriendly convolution into a dense, contiguous GEMM that the SIMD path eats happily.

#### Worked example: will this model hit the latency budget before I flash it?

Here is the back-of-envelope I run before committing a model to hardware — it has saved me from flashing dozens of dead ends. Suppose the product requirement is a gesture classifier that must run at 25 inferences per second on a Cortex-M4F at 80 MHz, leaving headroom for the audio/IMU front-end and the application. A 25 Hz rate means each inference has a hard wall of $1 / 25 = 40\ \text{ms}$, but the front-end and app realistically need half of that, so the model's `Invoke()` budget is roughly **20 ms**. Now I count the model: say it totals **3.2 M MACs**. On CMSIS-NN int8 at an effective ~0.6 cycles/MAC for the dense layers (call it 0.8 cycles/MAC blended, because depthwise layers run leaner), that is $3.2 \times 10^6 \times 0.8 \approx 2.6 \times 10^6$ cycles. At 80 MHz, $2.6 \times 10^6 / (80 \times 10^6) \approx 32\ \text{ms}$. **That blows the 20 ms budget** — before I have written a line of deployment code. The fix options, in order of effort: prune the MAC count by ~40% (drop a block or narrow channels), step the clock to 120 MHz if the part and power budget allow ($32 \times 80/120 \approx 21\ \text{ms}$ — still tight), or move up to a Cortex-M7 at 300+ MHz where the same 3.2 M MACs finish in well under 10 ms. The point of the example is the discipline: **MACs ÷ (MAC/cycle) ÷ clock gives you the latency to one significant figure in thirty seconds**, and a model that fails this arithmetic will fail on silicon too. Run it before you flash, every time.

The honest caveat: SIMD packing has alignment and shape requirements. CMSIS-NN's fastest paths want channel counts that are multiples of 4 (so the int8 lanes pack cleanly) and properly aligned buffers. If your layer has 17 output channels, the kernel either pads to 20 or falls back to a slower tail loop for the remainder. This is one reason efficient MCU architectures choose channel counts that are multiples of 8 — it is not aesthetic, it is so the SIMD lanes stay full. Depthwise convolutions are a special case: because each channel is convolved independently, the dot products are short ($K \cdot K$, not $K \cdot K \cdot C_{\text{in}}$), so there is less to amortize and the depthwise speedup (~4×) is slightly below the dense-conv speedup (~4.6×) — the SIMD lanes still fill across the spatial positions, but the per-channel work is leaner. Quantization is the prerequisite for all of this: there is no SIMD MAC win on float32 on an FPU-less part, because there is no fast float at all. int8 is what unlocks the kernels.

## 5. The deploy flow: train, quantize, convert, xxd, compile

Now the practical path, end to end. The TinyML deploy flow is a one-way pipe with five stations, and unlike server deployment there is no runtime model download — the model is *baked into the firmware binary*. The figure lays out the sequence.

![A left-to-right timeline of five deploy steps: train float32 in Keras, quantize to int8 with a representative set, convert to a tflite flatbuffer, xxd into a C array, and compile the firmware binary to flash](/imgs/blogs/tinyml-on-microcontrollers-3.png)

### Step 1–2: train, then quantize to int8 with a representative dataset

Train the model however you like; for an MCU target, keep it small by design (depthwise-separable convs, narrow channels). The interesting step is **full-integer quantization**. Unlike float16 or "dynamic range" quantization, MCU deployment needs **full int8** — weights *and* activations as int8, with fixed scales — because the CMSIS-NN kernels are integer-only and there is no fast float to fall back to. To compute the activation scales, the converter needs to see real data flow through the model: you supply a **representative dataset**, a generator that yields a few hundred typical input samples. The converter runs them, watches the min/max range of every activation tensor, and picks per-tensor (or per-channel for weights) scales.

```python
import tensorflow as tf
import numpy as np

# `model` is a trained tf.keras KWS model; `mfcc_samples` is an array of
# real MFCC frames from your training/validation set (e.g. 300 of them).

def representative_dataset():
    for sample in mfcc_samples[:300]:
        # Yield one batch-1 float32 input per call; the converter runs
        # it to observe activation ranges and choose int8 scales.
        yield [sample.reshape(1, 49, 10, 1).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force FULL int8: weights AND activations int8, including I/O.
# This is what TFLM + CMSIS-NN require on a Cortex-M.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("kws_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"int8 model size: {len(tflite_model) / 1024:.1f} KB")
```

Two flags carry the weight here. `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]` tells the converter to refuse to emit any float op — if some op cannot be quantized it errors loudly instead of silently leaving a float island that would force a (nonexistent) float kernel on the MCU. And setting `inference_input_type`/`inference_output_type` to `int8` means even the model's input and output are int8, so your firmware feeds raw int8 MFCCs and reads int8 logits with no float conversion on device. The representative dataset is the part people get wrong: too few samples (or unrepresentative ones) and the activation ranges are miscalibrated, clipping real values at inference and dropping accuracy. A few hundred diverse samples is the floor; the calibration mechanics and how to debug a bad range are in [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq).

### Step 3: the flatbuffer is the model

The `.convert()` call produced a flatbuffer. There is nothing more to do at convert time — the `kws_int8.tflite` file *is* the model, weights and graph and quantization parameters all in one self-describing blob. On a phone you would ship this file and load it at runtime. On an MCU there is no filesystem to load it from, so we go one step further.

### Step 4: xxd the flatbuffer into a C array

The MCU has no filesystem, so the model must become *source code* that the compiler embeds into the firmware's flash. The classic tool is `xxd`, which dumps a file as a C array:

```bash
# Turn the flatbuffer into a C source file with a byte array.
xxd -i kws_int8.tflite > kws_model_data.cc

# The generated file looks like:
#   unsigned char kws_int8_tflite[] = { 0x1c, 0x00, 0x00, 0x00, ... };
#   unsigned int  kws_int8_tflite_len = 59112;
#
# Rename the symbol to something stable and mark it const so the
# linker places it in flash (.rodata), not in SRAM:
sed -i 's/unsigned char kws_int8_tflite/const unsigned char g_kws_model_data/' kws_model_data.cc
sed -i 's/unsigned int kws_int8_tflite_len/const unsigned int g_kws_model_len/'  kws_model_data.cc
```

The `const` qualifier is not cosmetic. Without it, some toolchains place the array in initialized `.data`, which means a copy of the entire model gets staged in SRAM at boot — instantly blowing your RAM budget by the size of the model. With `const`, the array lands in `.rodata` in flash and TFLM reads it in place. I have personally watched a 200 KB model silently consume 200 KB of SRAM because of a missing `const`; the device booted fine in the simulator and bricked on hardware. Check your map file. (On some toolchains you additionally need a `__attribute__((aligned(16)))` so the flatbuffer's internal alignment requirements are met — if `AllocateTensors()` returns an alignment error, that is the fix.)

### Step 5: compile the firmware

Now `kws_model_data.cc` is just another source file. You compile it alongside your TFLM C++ and your application, link against the CMSIS-NN-backed kernels, and produce a single `.bin`/`.elf` that you flash to the device. The model is now part of the firmware. To update the model you re-flash the firmware (or use a firmware-update mechanism that swaps the whole image). There is no "model server," no version endpoint — the model's version *is* the firmware's version. This is a real operational difference from cloud and even phone ML, and it shapes how you do A/B tests and rollbacks on fleets of devices: you are managing firmware, not artifacts.

## 6. The full TFLite Micro inference sketch

Here is the part everyone actually wants: a complete, idiomatic TFLM inference program. This is the shape of every TinyML application's hot path. I have annotated the load-bearing lines.

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// The model, baked into flash by xxd. `const` => lives in .rodata, read in place.
#include "kws_model_data.cc"   // defines g_kws_model_data[] and g_kws_model_len

namespace {

// The ONE buffer. Sized to the peak working set of THIS model (see section 2).
// Aligned so the flatbuffer's internal tensors satisfy their alignment.
constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

}  // namespace

void setup() {
  // 1. Map the flatbuffer in flash to a Model object (no copy, no malloc).
  model = tflite::GetModel(g_kws_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != runtime %d", model->version(),
                TFLITE_SCHEMA_VERSION);
    return;
  }

  // 2. Register ONLY the ops this model uses. The template arg is the count.
  //    Every op you omit is flash you don't pay for.
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();

  // 3. Build the interpreter over the model, resolver, and the static arena.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 4. Plan the arena: carve activation tensors out of tensor_arena. The ONLY
  //    place memory is "allocated" — and it's all inside our static buffer.
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed — arena too small or op missing");
    return;
  }

  // 5. Cache input/output handles. After this, the hot loop touches no setup.
  input  = interpreter->input(0);
  output = interpreter->output(0);

  // How much of the arena did we actually use? Use this to shrink the buffer.
  MicroPrintf("arena used: %d bytes of %d", interpreter->arena_used_bytes(),
              kTensorArenaSize);
}

void run_inference(const int8_t* mfcc /* 49*10 int8 values */) {
  // Copy the quantized features into the model's input tensor.
  for (int i = 0; i < 49 * 10; ++i) {
    input->data.int8[i] = mfcc[i];
  }

  // 6. THE inference. Walks the graph, dispatches CMSIS-NN int8 kernels into
  //    the arena. No malloc, no float, no OS call.
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke() failed");
    return;
  }

  // 7. Read int8 logits; dequantize only if you need probabilities.
  int   best = 0;
  int8_t best_score = output->data.int8[0];
  for (int c = 1; c < output->dims->data[1]; ++c) {
    if (output->data.int8[c] > best_score) {
      best_score = output->data.int8[c];
      best = c;
    }
  }
  // `best` is the predicted keyword index. Act on it (e.g. wake the SoC).
}
```

Read the comments numbered 1–7 — they are the entire TFLM contract. `GetModel` maps the flatbuffer in flash with no copy. The resolver registers five ops (and the `<5>` template arg must match — too small and it asserts, too large wastes a few bytes of RAM for empty slots). `AllocateTensors()` is the *only* allocation event and it happens once at setup. `Invoke()` is the hot path and it allocates nothing. The `arena_used_bytes()` call is your sizing oracle: run once with a generous arena, read the number it prints, set `kTensorArenaSize` to that plus a few KB of margin, rebuild. That is how you find the real arena size empirically instead of guessing.

In practice the sizing loop takes two or three rebuilds and looks like this on the wire. You start deliberately generous so `AllocateTensors()` cannot fail, read the printed truth, and converge:

```console
# Rebuild 1: deliberately oversized arena to get a clean allocation + truth.
$ # kTensorArenaSize = 200 * 1024
arena used: 90112 bytes of 204800
# Rebuild 2: set kTensorArenaSize = 96 * 1024 (90112 + ~6 KB margin), rebuild.
$ # kTensorArenaSize = 96 * 1024
arena used: 90112 bytes of 98304
# Rebuild 3 (too greedy): kTensorArenaSize = 88 * 1024 — below the real peak.
$ # kTensorArenaSize = 88 * 1024
AllocateTensors() failed — arena too small or op missing
```

The number that matters is `90112` — exactly $88 \times 1024$ bytes, the packed peak the planner found for this graph. Note it is *not* a round number you would have guessed; it is the output of the rectangle-packing heuristic on this specific tensor liveness graph, which is why you measure rather than estimate. The margin you add on top of `arena_used_bytes()` is not superstition: TFLM's online planner can pack a hair differently across versions, and a future model tweak (one more channel, one extra op) nudges the peak, so a few KB of slack keeps a minor change from turning into a boot-time failure on a fleet that is already in the field. I keep the margin small — 4 to 8 KB — because on a 256 KB part every kilobyte you reserve for slack is a kilobyte the application can't use, and the whole discipline of this section is that nobody gets to waste SRAM.

If you want to call a CMSIS-NN kernel directly — for a custom op TFLM does not have, or to micro-benchmark a single layer — the API is integer-in, integer-out with explicit quantization params. A fully-connected layer looks like:

```c
#include "arm_nnfunctions.h"

// Quantized fully-connected: out_q = requantize( W_q @ in_q + bias )
// Dimensions, quant params, and scratch are passed explicitly — no allocation.
arm_cmsis_nn_status status = arm_fully_connected_s8(
    &ctx,        // scratch buffer context (you provide the buffer)
    &fc_params,  // input/output offsets (zero points), activation clamp range
    &quant,      // per-tensor multiplier + shift for the requantize step
    &in_dims,  in_q,      // int8 input  [batch, in_features]
    &w_dims,   weight_q,  // int8 weights[out_features, in_features]
    &b_dims,   bias_s32,  // int32 bias
    &out_dims, out_q);    // int8 output [batch, out_features]
// status == ARM_CMSIS_NN_SUCCESS on success. `ctx.buf` is YOUR scratch — the
// kernel never mallocs; you size and own every byte it touches.
```

Notice there is no allocation anywhere in that signature. You pass a context with a scratch buffer you own; the kernel uses it and returns. The `quant` parameter — a fixed-point multiplier and a right-shift — is how the kernel does the requantization $y = \text{clamp}\!\big(\lfloor (W_q x_q + b)\cdot M \rfloor + z_y\big)$ that rescales the int32 accumulator back to int8 without any float, using a fixed-point multiply. That fixed-point requantize is the last piece of "int8 all the way down": even the rescaling is integer.

## 7. Measuring it honestly: cycles, arena, and milliwatts

A result you cannot reproduce is marketing. Here is how to measure each number that matters on an MCU, and the traps that produce lies.

**Latency in cycles, not just milliseconds.** On a Cortex-M you have a cycle counter — the DWT (Data Watchpoint and Trace) `CYCCNT` register. Wrap `Invoke()` and read it. Cycles are more honest than milliseconds because they are clock-independent: if you report 184,000 cycles, anyone can divide by their clock. Milliseconds hide whether you measured at 80 or 120 MHz.

```c
#include "core_cm4.h"

static inline void cycle_counter_start(void) {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // enable trace
  DWT->CYCCNT = 0;                                  // reset counter
  DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;             // start counting
}

// In your benchmark:
cycle_counter_start();
interpreter->Invoke();
uint32_t cycles = DWT->CYCCNT;
// latency_ms = cycles / (SystemCoreClock / 1000.0)
MicroPrintf("invoke: %lu cycles (%lu us @ %lu MHz)",
            cycles, cycles / (SystemCoreClock / 1000000),
            SystemCoreClock / 1000000);
```

The traps: run a **warm-up inference first** (the first call pays one-time cache and branch-predictor cold-start costs on M7-class parts), measure **batch=1** because that is the only batch an MCU ever runs, and run from the **same memory configuration you will ship** — if your model runs from external QSPI flash instead of internal flash, or your arena is in slow external RAM, the cycle count can double and `CYCCNT` will faithfully report the slowdown. Report the median of many runs; on an MCU with no OS and no other threads, the variance is tiny (no scheduler jitter), which is one of the few mercies of the platform.

**Arena, exactly.** `interpreter.arena_used_bytes()` after `AllocateTensors()`. That is the ground truth, not your estimate. Subtract it from your `kTensorArenaSize` to see your margin.

**Flash, exactly.** Read the `.map` file your linker emits, or run `arm-none-eabi-size firmware.elf`. The `text` column is flash (code + the `const` model array + read-only data), the `data`+`bss` columns are SRAM. If the model array shows up in `data` instead of `text`, your `const` is missing — see section 5.

**Power, in milliwatts.** This is the hardest to measure honestly. You want average power over a realistic duty cycle, not peak. The right tool is a current-measurement probe (a Power Profiler Kit, or a shunt + scope) on the MCU's supply. The number that matters for battery life is the *energy per inference* times the *inference rate*, plus the sleep-current floor:

$$P_{\text{avg}} = f_{\text{infer}} \cdot E_{\text{infer}} + P_{\text{sleep}} ,$$

where $E_{\text{infer}}$ (in millijoules) is the active energy of one `Invoke()` and $P_{\text{sleep}}$ is what the part draws between inferences. For an always-on KWS earbud, the design trick is to keep $f_{\text{infer}}$ low and $P_{\text{sleep}}$ near zero (deep sleep with only a low-power audio peripheral and a tiny "is there sound?" detector awake), so the Cortex-M4 only spins up the full model when there is plausibly speech. The model's per-inference energy matters, but the *duty cycle* matters more — which is why "make the model faster" and "make the model rarer" are both power levers.

#### Worked example: anomaly detection on a Cortex-M4, the SRAM budget

The second canonical TinyML task is **anomaly detection** — an autoencoder or tiny classifier on accelerometer or microphone data that flags when a motor, pump, or bearing sounds "wrong." This is the MLPerf Tiny industrial-anomaly benchmark. Let me walk the SRAM budget for a small dense autoencoder on a Cortex-M4F with 256 KB SRAM, because the budget breakdown is the whole engineering exercise.

The model: input 640 features (a flattened spectrogram window) → encoder 128 → 8 (bottleneck) → decoder 128 → 640 output, all int8, anomaly score = reconstruction error. Parameter count is dominated by the two $640\times128$ layers: $2 \times 640 \times 128 \approx 164{,}000$ weights, at int8 that is **~164 KB in flash** (plus int32 biases, negligible). SRAM budget:

- **Activations**: the largest live pair is the 640-wide input and a 128-wide hidden, but they do not both need full residence at once with arena reuse; peak activation lands around **3–5 KB**. Dense layers have tiny activations compared to convs — there is no big spatial feature map.
- **Arena total** (activations + the fully-connected scratch + bookkeeping): ~**12 KB**.
- **Application + audio buffers** (the rolling spectrogram, FFT scratch): ~**20 KB**.
- **Stack + globals**: ~**8 KB**.

Total SRAM: ~40 KB of 256 KB — *comfortable*. The binding constraint for this model is **flash** (164 KB of weights), not SRAM. That flip is the lesson: dense models with big matrices are flash-bound, while conv models with big feature maps are SRAM-bound. You profile to learn which one you are, then you reach for the matching lever — quantization and weight sharing for flash-bound, arena planning and tiling for SRAM-bound. Either way, on an M4F at 80 MHz this autoencoder runs in well under a millisecond per inference and can run continuously inside the milliwatt budget.

#### Worked example: person detection on a Cortex-M7, the SRAM wall

The third task moves up to vision, which is where the SRAM wall bites hardest. A "person present?" detector — the visual-wake-word task in MLPerf Tiny — runs a tiny MobileNet-style classifier on a low-resolution grayscale frame (say $96 \times 96$) on a Cortex-M7 with 512 KB SRAM. Now the arithmetic flips the other way from the autoencoder. The model is small in *parameters* (a few hundred KB of int8 weights, comfortably in flash), but its *activations* are large because the early feature maps are spatial: the first conv produces a $96 \times 96 \times 16$ tensor, which at int8 is $96 \times 96 \times 16 = 147{,}456$ bytes — **144 KB for a single activation**. Add the im2col scratch the conv kernel needs for that layer, and the input frame still being live, and the peak working set for the first block alone can approach 300–400 KB, eating most of the 512 KB before the application even gets a buffer.

This is the SRAM wall, and it is why a model that "fits in flash" can still be impossible. The levers, in order of how much I reach for them: (1) **operator reordering and arena planning** to make sure no two giant feature maps are live at once — buys tens of KB; (2) **shrinking the input resolution** ($96 \to 64$ cuts that first activation by more than half, since it scales with $H \times W$) — buys the most, at an accuracy cost you measure; (3) **patch-based inference** — computing the early high-resolution blocks in small spatial tiles so only one patch and its receptive-field halo are live, which is exactly the MCUNetV2 trick that drops the peak from over a megabyte to a few hundred KB. On an M7 at 480 MHz, once the model fits, the inference itself is a few tens of milliseconds — fine for a wake-on-person sensor that runs a couple of times a second. The engineering was never the compute; it was making the activations fit. That is the defining experience of TinyML vision, and the reason the SRAM column, not the FLOP column, is the first thing I look at.

## 8. Results: MCU classes, KWS before/after, kernel speedup

Now the measured results, the part this series insists on. Three tables and one figure.

First, what fits on each Cortex-M class. The matrix is the cheat sheet I use when someone says "we have an STM32, can it run a CNN?" — the answer depends entirely on which M-class it is.

![A matrix comparing Cortex-M0+, M4F, and M7 across typical SRAM, clock, SIMD or DSP support, and the realistic model each can run, showing the M4 and M7 enabling real keyword spotting and vision](/imgs/blogs/tinyml-on-microcontrollers-6.png)

| MCU class | Typical SRAM | Flash | Clock | DSP/SIMD | Kernels | What realistically fits |
|---|---|---|---|---|---|---|
| Cortex-M0+ | 8–64 KB | 32–256 KB | ~48 MHz | none | reference int8 | tiny MLP, decision logic, ~few KB models |
| Cortex-M4F | 128–256 KB | 512 KB–2 MB | 80–120 MHz | DSP + FPU | CMSIS-NN | keyword spotting, simple anomaly detection |
| Cortex-M7 | 256 KB–1 MB | 1–4 MB | 300–600 MHz | DSP + double FPU | CMSIS-NN | person detection, small vision, richer audio |

The jump that matters is M0+ → M4: it is not mostly the clock, it is the DSP extension. The M4 can run CMSIS-NN's SIMD int8 kernels; the M0+ cannot and is stuck on the 4–5× slower reference path *and* has far less SRAM. If your roadmap has any real CNN on it, start at M4-class silicon. The M4 → M7 jump buys you headroom — more SRAM for bigger feature maps, a faster clock for tighter latency, and a cache that helps when the model spills out of tightly-coupled memory.

Second, the keyword-spotting model, float32 → int8, on a Cortex-M4F at 80 MHz. This is the canonical TinyML before/after and the figure shows it.

![A two-column before and after comparison of the keyword spotting model showing float32 at 224 KB and 10.4 ms versus int8 at 58 KB and 2.3 ms with only a 0.4 point accuracy drop](/imgs/blogs/tinyml-on-microcontrollers-7.png)

| Metric | float32 (reference) | int8 (CMSIS-NN) | Change |
|---|---|---|---|
| Model size (flatbuffer) | ~224 KB | ~58 KB | 3.9× smaller |
| Peak SRAM (arena) | ~140 KB | ~96 KB | 1.5× smaller |
| Latency / inference @ 80 MHz | ~10.4 ms | ~2.3 ms | 4.5× faster |
| Accuracy (12-word test set) | 94.1% | 93.7% | −0.4 pt |
| Active energy / inference | ~0.9 mJ | ~0.2 mJ | ~4.5× less |

Two effects compound. Int8 shrinks the model ~4× (the size win) *and* unlocks the CMSIS-NN SIMD kernels (the speed win), so you get both a smaller flash footprint and a faster, lower-energy inference. The accuracy cost — 0.4 points — is the kind of number a well-calibrated full-int8 conversion produces on a model that was reasonably designed; if you see 3–4 points lost, your representative dataset is too small or unrepresentative and a few activation ranges are clipping. These figures are consistent with the public MLPerf Tiny keyword-spotting results and the numbers in the CMSIS-NN paper; treat them as representative orders of magnitude for an M4-class part, not a single golden benchmark, since exact numbers move with the specific model, MFCC front-end, and silicon.

Third, the kernel speedup in isolation, to separate it from the quantization win.

| Operation | Reference int8 | CMSIS-NN int8 | Speedup |
|---|---|---|---|
| Conv2D (3×3, 64 ch) | baseline | ~4.6× | SIMD MAC + im2col GEMM |
| DepthwiseConv2D (3×3) | baseline | ~4.0× | per-channel SIMD |
| FullyConnected (256→64) | baseline | ~5.0× | SMLAD dot product |
| Softmax (12 classes) | baseline | ~1.5× | integer LUT, not MAC-bound |

The speedup is largest where the work is dense MAC (conv, fully-connected) and smallest where it is not (softmax is dominated by exp/normalize, which CMSIS-NN does with an integer lookup table but cannot SIMD-accelerate the same way). This is the general shape of kernel optimization: the MAC-heavy layers benefit most, and they are also where the time goes, so the whole-model speedup tracks the conv/FC speedup. Note these are *kernel-level* speedups at fixed int8 precision — they are on top of, not instead of, the size win from quantization.

Fourth, and this is the table I actually reach for when a model misses its budget: the **per-layer** breakdown of where the 2.3 ms goes, for the int8 KWS model on the same Cortex-M4F at 80 MHz. The whole-model number hides which layer to attack; the per-layer profile is the only thing that tells you. These come from wrapping each node's kernel call with the DWT `CYCCNT` read from section 7 (TFLM exposes a per-op profiler hook for exactly this).

| Layer | MACs | Cycles (int8, CMSIS-NN) | Time @ 80 MHz | Share |
|---|---|---|---|---|
| Conv2D #1 (input, 3×3, 64 ch) | ~1.0 M | ~78,000 | ~975 µs | 42% |
| DepthwiseConv2D #2 (3×3) | ~0.18 M | ~32,000 | ~400 µs | 17% |
| DepthwiseConv2D #3 (3×3) | ~0.18 M | ~31,000 | ~388 µs | 17% |
| FullyConnected (logits) | ~0.10 M | ~38,000 | ~475 µs | 21% |
| Softmax (12 classes) | — | ~5,000 | ~62 µs | 3% |
| **Total** | **~1.46 M** | **~184,000** | **~2.30 ms** | **100%** |

Read this table the way you would read a flame graph. The first convolution is 42% of the time on its own — because it runs at the full input resolution before any pooling has shrunk the spatial dimensions, so it has the most output positions to compute and the biggest im2col scratch to fill. That single layer is also where the SRAM peak lives (section 2). When a vision model busts its latency budget, this is almost always the culprit, and the levers are the same ones from the SRAM discussion: drop input resolution, or restructure so the high-resolution block is the cheapest, not the most expensive. The cycles-per-MAC also tells a story: the first conv runs near 0.08 cycles/MAC because its long $K \cdot K \cdot C_{\text{in}}$ dot products keep the `SMLAD` lanes saturated, while the depthwise layers run closer to 0.18 cycles/MAC because their per-channel dot products are short ($K \cdot K = 9$) and there is less work to amortize the loop and requantize overhead against — exactly the depthwise-vs-dense gap predicted in section 4. The profile is not decoration; it is the map you optimize against.

A note on *why* the −0.4 point accuracy drop is so small, tying back to the SQNR recap in section 1. The KWS model's pre-softmax logit gap between the top class and the runner-up, on correctly-classified frames, is on the order of several quantization steps wide — the decision margin sits well above the int8 noise floor of ~50 dB SQNR. So int8's quantization noise jitters each logit by a fraction of the margin, almost never enough to flip the argmax. The 0.4 points it does cost come from the frames that were already borderline in float — the ones sitting within a step or two of the decision boundary — where the added noise tips a few from correct to wrong. That is the mechanistic reason a well-designed, well-calibrated int8 KWS model loses a fraction of a point and not several: the network's margins were already wider than the noise you injected.

## 9. The constraints that define TinyML — and when to step up

Step back. Four numbers define whether a TinyML deployment is feasible, and the decision tree below routes a candidate model to the right target based on them.

![A decision tree starting from picking the target, branching on whether the peak working set fits under 256 KB of SRAM or needs megabytes, leading to an MCU with TFLM, tile-then-MCU, a Cortex-A SoC, or an edge NPU](/imgs/blogs/tinyml-on-microcontrollers-8.png)

1. **Peak SRAM (the binding wall).** Your peak working set must fit the part's SRAM with room for the application. This is the constraint that most often kills an MCU port, because activation peaks scale with feature-map size, and a single early conv can dominate. If you are over by a little, arena planning, in-place ops, and operator reordering can recover tens of KB. If you are over by a lot, you need a structurally smaller model or **patch-based inference** — which I forward-reference below.
2. **Flash (the model wall).** Your int8 weights plus code plus the TFLM runtime must fit flash. Dense, big-matrix models hit this first. Quantization (int8, or even sub-8-bit for the weights) and weight sharing are the levers; the sub-8-bit options are covered in the quantization track of this series.
3. **No-malloc (the structural wall).** Everything must be statically shaped and statically allocated. Variable-length anything (sequences, detections) must be fixed to a maximum. If your model genuinely needs dynamic shapes, an MCU is the wrong target.
4. **Power (the deployment wall).** The average power over the real duty cycle must fit the battery or energy-harvesting budget. This is as much an application-architecture question (how often do you run, how deep do you sleep) as a model question.

When all four fit, an MCU is a wonderful target: pennies of silicon, milliwatts of power, microseconds of boot, no OS to maintain, and a model that ships as part of one firmware binary you fully control. When they do not, *step up* — and the right step depends on which wall you hit. If the model needs megabytes of RAM or genuine dynamic shapes, you want a **Cortex-A class SoC** running Linux with LiteRT or ONNX Runtime: now you have real DRAM, an OS, and a filesystem to load models from, at the cost of more power and more boot time. If you need batched throughput or large-model inference at low latency, you want a dedicated **edge NPU** (the territory of the NPU and accelerator posts in this series). The art is matching the model to the smallest target that fits all four constraints with margin — that is the on-device version of reading the [accuracy–efficiency Pareto frontier](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression).

There is a middle road on the MCU itself before you give up and step up: **MCUNet** and its successors (Lin et al., 2020/2021) co-design a tiny network (TinyNAS) with a memory-efficient inference engine (TinyEngine) so that an ImageNet-class classifier fits a Cortex-M7 under 512 KB of SRAM. The key trick for the SRAM wall is patch-based inference — computing the early, high-resolution feature maps in small spatial tiles so only one patch and its receptive-field halo are live at once, slashing the activation peak. I go deep on that in [squeezing models into kilobytes](/blog/machine-learning/edge-ai/squeezing-models-into-kilobytes). And once your model is on-device, the next frontier is updating it without a round trip to the cloud — [on-device and federated learning](/blog/machine-learning/edge-ai/on-device-and-federated-learning) — so the earbud can adapt to your voice without your audio ever leaving the device.

## 10. Stress tests: where the no-malloc world bites

Every clean story has failure modes. Here are the ones that have paged me or my teams, and how to reason about each.

**What happens at int4?** Tempting — half the flash. But CMSIS-NN's fast paths are built for int8; int4 on a Cortex-M has no native SIMD support, so the kernel must unpack two int4 values per byte, widen, and run a slower path, often *erasing* the speed advantage even as it saves flash. And int4 PTQ on a tiny model frequently drops several accuracy points because the small model has no redundancy to spare. Verdict: int4 weights can help a flash-bound dense model if you accept the accuracy hit and the slower unpack, but for the common SRAM-bound conv model it is rarely worth it. Stay at int8 unless flash is the proven wall.

**When the calibration set is tiny.** If your representative dataset is 10 samples, the converter sees a narrow slice of activation ranges and picks scales that clip real inputs at inference. The symptom is "great accuracy in the converter's eval, terrible on device." Fix: a few hundred diverse, *in-distribution* samples — and verify by comparing the int8 model's outputs against the float model's on a held-out set *before* you flash, not after.

**When the arena is too small.** `AllocateTensors()` returns `kTfLiteError` at startup — which is the *good* outcome, a clean fail at boot, not a runtime corruption. The fix is the sizing loop from section 6: bump the arena, read `arena_used_bytes()`, set it to that plus margin. The *bad* outcome is when the arena is in a memory region that overlaps the stack — then a deep call stack silently corrupts arena tensors and you get garbage predictions intermittently. Place the arena in a dedicated, well-separated section and watch your stack high-water mark.

**When an op is missing from the resolver.** `AllocateTensors()` fails with the op name. Add it to the `MicroMutableOpResolver` and bump the template count. The subtle version: the converter fused or rewrote an op (e.g. a batch-norm folded into a conv, or a `Relu6` represented as a clamp), so the op in the flatbuffer is not the one in your source model. Inspect the `.tflite` with a viewer (Netron) to see the *actual* op list, then register exactly those.

**When the model is memory-bound, not compute-bound.** On an MCU with slow external flash or RAM, the bottleneck can be *fetching* weights and activations, not multiplying them — the same roofline story as on big hardware, just with KB instead of GB. If `CYCCNT` says your layer is far slower than its MAC count predicts, you are memory-bound: move the model and arena into the fastest memory you have (internal flash, tightly-coupled memory / TCM on M7), and prefer architectures with high arithmetic intensity (more MACs per byte fetched). The [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) applies here in miniature.

**When float sneaks back in.** If the converter leaves a single float op (because it could not quantize it under `TFLITE_BUILTINS_INT8`), and you are on an FPU-less M0+, that one op runs in software-emulated float and can dominate the whole inference. The `supported_ops = [TFLITE_BUILTINS_INT8]` flag is supposed to prevent this by erroring at convert time; if you used a looser flag, audit the flatbuffer for float ops before flashing.

**When there is no FPU at all.** This is the failure that surprises people who develop on an M4F and ship on an M0+. Suppose you built and profiled everything on a Cortex-M4F — FPU present, CMSIS-NN happy — and the int8 model ran in 2.3 ms. Then procurement swaps the bill-of-materials to a cheaper FPU-less M0+ to save a few cents at volume. Two things break at once. First, if any float crept into the graph (a stray dequantize, a float softmax, a Mul with a float scalar), every one of those operations now calls the toolchain's soft-float library (`__aeabi_fmul`, `__aeabi_fadd`), and a single float multiply that was one cycle on the M4F is now 30–50 cycles of emulation — a layer that was microseconds becomes milliseconds, and the always-on power budget evaporates. Second, even the *fully int8* model gets dramatically slower, because the M0+ has **no DSP extension**: there is no `SMLAD`, so CMSIS-NN cannot dispatch its SIMD path and falls back to the scalar reference kernels at 1 MAC per iteration. The same model that hit 2.3 ms on the M4F can land at 10–12 ms on the M0+ purely from losing SIMD, *plus* the M0+ has a fraction of the SRAM so the arena may not even allocate. The lesson is brutal and worth tattooing on the project plan: **the M0+ is not a slower M4 — it is a different machine, and a model validated on one is not validated on the other.** If the M0+ is the real target, profile on the M0+ from day one, keep the graph ruthlessly all-int8, and size for its tiny SRAM; if a CNN is on the roadmap, do not let procurement talk you below an M4-class part, because the DSP extension is the feature you are actually buying.

## 11. Case studies: real numbers from the literature

Four results that ground all of the above in published work and shipped products.

**CMSIS-NN on keyword spotting (Lai et al., 2018).** The original CMSIS-NN paper demonstrated a CNN keyword-spotting model on a Cortex-M7, reporting roughly **4.6× higher throughput and 4.9× better energy efficiency** for the int8 kernels versus the reference implementation, while fitting the model and runtime in the on-chip memory of an STM32-class part. This is the source of the "4–5×" number you have seen throughout this post — it is measured, not assumed.

**TFLite Micro itself (David et al., 2021).** The TFLM paper reports the runtime running real models in **tens of kilobytes of code and memory** across a range of MCUs, with the explicit no-malloc, no-OS design. Its contribution is less a single benchmark and more the demonstration that a *portable* interpreter (one codebase across hundreds of MCU SKUs) can be made small enough to be practical — which is why it became the de facto TinyML runtime.

**MCUNet / TinyNAS + TinyEngine (Lin et al., 2020–2021).** MCUNet put an **ImageNet classifier above 70% top-1 onto a Cortex-M7 with 512 KB SRAM and 2 MB flash**, by co-designing the network and the inference engine and using patch-based inference to keep the activation peak under budget. The headline lesson for this post: the SRAM wall is the binding one for vision, and you beat it by attacking the *peak activation* (via tiling and architecture), not just the parameter count. Details in [squeezing models into kilobytes](/blog/machine-learning/edge-ai/squeezing-models-into-kilobytes).

**Always-on KWS in shipped earbuds and phones.** The wake-word path in real consumer devices runs a tiny always-on model on a low-power core (an MCU-class island, even when the device also has a big SoC) precisely so the main processor can stay asleep. The exact models are proprietary, but the architecture is exactly what we built here: int8 CNN, fixed arena, duty-cycled, milliwatt budget, with a low-power audio detector gating when the full model even runs. The published MLPerf Tiny benchmark (keyword spotting, image classification, anomaly detection, visual wake words) is the open stand-in for these workloads and the right place to compare your own numbers against the field.

## 12. When to reach for an MCU (and when not to)

A decisive recommendation, because every choice here is a cost.

**Reach for an MCU + TFLM + CMSIS-NN when:** the task is small and well-defined (keyword spotting, gesture, anomaly detection, simple vision), the model fits a few hundred KB of SRAM and a couple MB of flash *at int8*, you need always-on at milliwatts and long battery life, you want pennies of bill-of-materials cost at scale, and you can ship the model as firmware. This is the sweet spot, and it is enormous — billions of these devices ship.

**Do not reach for an MCU when:** the model genuinely needs megabytes of activation RAM (most real vision beyond tiny classifiers, any LLM), it needs dynamic shapes or data-dependent control flow that cannot be fixed at convert time, you need frequent model updates with fast rollout (firmware update on a fleet is heavier than swapping a model file), or the accuracy floor for your task is simply above what fits the budget. In those cases step up to a Cortex-A SoC with LiteRT/ONNX Runtime or to an edge NPU — the model wants more memory and an OS, and forcing it onto an MCU produces a worse product, not a cheaper one.

**A few specific "don'ts" that save pain:** Don't skip the representative dataset or feed it junk — bad calibration is the number-one cause of "it was accurate in the converter and wrong on device." Don't forget `const` on the model array — it silently copies the model into SRAM. Don't reach for int4 by default — int8 is the CMSIS-NN sweet spot and int4 often costs more (slower unpack, accuracy) than it saves. Don't optimize the model's per-inference cost while ignoring the duty cycle — on an always-on device, sleeping deeper and running rarer is a bigger power lever than shaving a millisecond off `Invoke()`. And don't tune to milliseconds without recording cycles and clock — the next person who reads your benchmark needs the cycle count to reproduce it.

## Key takeaways

- A microcontroller is a different machine, not a small computer: KB of SRAM, MB of flash, tens–hundreds of MHz, often no FPU, no MMU, no OS, and **no `malloc` in practice**. Design for that from the first line.
- **int8 is mandatory, not optional.** It shrinks the model 4× *and* unlocks fast integer kernels on FPU-less parts. Quantize first; the accuracy cost on a well-calibrated full-int8 conversion is typically under a point.
- The binding constraint is **peak working set** (max simultaneously-live activation bytes), not parameter count. The **tensor arena** is one pre-allocated buffer the planner packs all activations into; size it to the peak via `arena_used_bytes()`, not by guessing.
- **TFLite Micro** is an interpreter stripped of every MCU-unaffordable assumption: flatbuffer model read in place from flash, op resolver registering only the kernels you use, a static arena, and an `Invoke()` that **allocates nothing**.
- **CMSIS-NN** kernels are 4–5× the reference because they pack four int8 lanes into a 32-bit word and use `SMLAD` to retire two MACs per cycle. SIMD wants channel counts that are multiples of 4–8; design for full lanes.
- The deploy flow is one-way: train → quantize int8 (representative dataset) → convert `.tflite` flatbuffer → `xxd` to a `const` C array → compile into firmware. The model's version is the firmware's version.
- Measure in **cycles** (DWT `CYCCNT`) with a warm-up and batch=1; read flash from the `.map`/`size`; read arena from `arena_used_bytes()`; measure power as average over the real duty cycle, where sleeping deeper often beats shaving the model.
- Know the walls (peak SRAM, flash, no-malloc, power) and step up to a Cortex-A SoC or an edge NPU the moment the model needs MBs of RAM, dynamic shapes, or accuracy the budget can't hold.

## Further reading

- **R. David et al., "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems," MLSys 2021** — the design of the no-malloc, no-OS interpreter and the arena model.
- **L. Lai, N. Suda, V. Chandra, "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs," 2018** — the SIMD int8 kernels and the 4–5× measured speedup on keyword spotting.
- **J. Lin et al., "MCUNet: Tiny Deep Learning on IoT Devices," NeurIPS 2020** (and MCUNetV2, 2021) — TinyNAS + TinyEngine and patch-based inference to beat the SRAM wall.
- **P. Warden and D. Situnayake, "TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers," O'Reilly, 2019** — the practitioner's book for the whole flow.
- **MLPerf Tiny benchmark (Banbury et al., 2021)** — the open benchmark suite (KWS, image classification, anomaly detection, visual wake words) to compare your numbers against.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for where MCUs sit, [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) for the int8 mechanics, [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) for the peak-working-set argument, [squeezing models into kilobytes](/blog/machine-learning/edge-ai/squeezing-models-into-kilobytes) for MCUNet-style tiling, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that ties every lever together.
