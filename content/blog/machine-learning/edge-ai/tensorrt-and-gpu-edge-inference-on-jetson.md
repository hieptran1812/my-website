---
title: "TensorRT and GPU edge inference on Jetson: builder, INT8, and engine plans"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to actually hit a Jetson's peak with TensorRT — the builder's fusion and kernel auto-tuning, INT8 calibration, dynamic-shape optimization profiles, and the engine portability rules that bite you in production."
tags:
  [
    "edge-ai",
    "model-optimization",
    "tensorrt",
    "jetson",
    "int8",
    "gpu",
    "inference",
    "efficient-ml",
    "nvidia",
    "quantization",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-1.png"
---

The benchmark that started this for me was embarrassing in the most useful way. We had a ResNet-50-class classifier running on a Jetson Orin Nano, served through ONNX Runtime with the CUDA execution provider, and it clocked 12.6 ms per image at batch 1. The data sheet said the chip could do far better, the product wanted under 5 ms, and everyone's first instinct was to argue for a bigger Jetson. Instead I spent one afternoon exporting the same ONNX file, running it through TensorRT with `--fp16`, and then with `--int8` and a few hundred calibration images. The fp16 engine came back at 6.1 ms. The int8 engine came back at 3.4 ms, with the top-1 accuracy down by half a point. Same model, same chip, same power budget — **3.7× faster** because the inference stack finally stopped leaving the silicon idle.

That is the whole pitch for TensorRT in one paragraph. On NVIDIA edge GPUs — the Jetson Orin family — and on NVIDIA edge servers, a naive ONNX-Runtime-CUDA run is *not* hitting the chip's peak. It is calling generic CUDA kernels, one per layer, in fp32, materializing every intermediate tensor to global memory, and never asking which of several possible kernel implementations is actually fastest on *this* exact GPU. TensorRT is the tool that asks all of those questions at build time and bakes the best answers into a compiled artifact. A 2–5× speedup over the naive run is the normal result, not a lucky one, and most of that comes from three moves: fusing layers so the GPU stops round-tripping intermediates through memory, auto-tuning kernels so each layer uses the fastest implementation for the target, and dropping precision to fp16 or int8 where the model can take it.

By the end of this post you will be able to take a trained model, export it to ONNX, build an optimized TensorRT engine with `trtexec` and with the Python builder API, calibrate it to int8 with a real calibration loop, give it dynamic shapes through an optimization profile, profile it honestly on a Jetson with `tegrastats` and Nsight, and — just as importantly — know when *not* to reach for TensorRT because the engine's non-portability or NVIDIA-only nature makes it the wrong tool. This is the compiler-and-runtime layer of the series' [four-lever frame](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression): quantization, pruning, distillation, and efficient architectures all *produce* a model, but TensorRT is how you make NVIDIA hardware actually run that model at the chip's roofline. It composes with all four levers rather than replacing them.

Figure 1 is the mental spine for everything below: TensorRT splits cleanly into an offline **build** that compiles an ONNX graph into a serialized **engine** for one specific GPU plus precision plus shape range, and a lightweight **runtime** that deserializes that engine and executes it. Keep that two-phase split in mind; almost every confusing thing about TensorRT — why builds are slow, why engines do not transfer between machines, why the same model has different latency on two "identical" boxes — falls out of the fact that the build does all the expensive, target-specific work once and the runtime just replays the plan.

![A left to right timeline showing the TensorRT flow from exporting ONNX through the builder compiling the graph, serializing the engine to disk, the runtime deserializing it, and finally fast inference](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-1.png)

This post assumes you know what a GPU is, what an ONNX graph is, and roughly what int8 quantization does (the affine map from floats to integers, scale and zero-point). If int8 is fuzzy, read [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) and [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) first — this post takes the numerics as given and focuses on how TensorRT *applies* them on real NVIDIA hardware. If you want the broader map of where TensorRT sits among other inference runtimes, the series' [hardware landscape post](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) covers the chips and the runtime comparison covers the alternatives. Here, we go deep on one specific thing: how to get peak inference out of an NVIDIA edge GPU.

## 1. What "the builder" actually is

The single most important concept in TensorRT is the **builder**, and the single most important thing to internalize is that it is a *compiler*. It takes a portable, hardware-agnostic graph — an ONNX file — and emits a hardware-specific binary — an **engine**, also called a **plan**. The engine is not portable. It is not a model in the usual sense. It is a serialized sequence of "call this exact CUDA kernel with these exact parameters, in this order, reusing this memory layout," tuned for one GPU architecture, one TensorRT version, one precision configuration, and one range of input shapes.

That framing matters because it dissolves a lot of confusion. People ask "why is my TensorRT build taking ten minutes when ONNX Runtime loaded the model instantly?" The answer is that ONNX Runtime did not compile anything — it interpreted the graph and dispatched generic kernels. TensorRT spent those ten minutes actually *benchmarking* candidate kernels on your GPU to find the fastest ones. People ask "why did the engine I built on my workstation fail to load on the Jetson?" Because a compiler that targets a specific ISA does not produce binaries that run on a different ISA, and the GPU architecture *is* the ISA here. Once you hold "builder = compiler, engine = compiled binary" firmly, the rest of TensorRT stops being magic.

The builder does four distinct jobs, and they are worth separating because they are the four sources of the speedup. Figure 2 lays them out as a stack. I will spend a section on each, but the one-line versions are: it **fuses** adjacent layers into single kernels so the GPU stops shuttling intermediate tensors through slow global memory; it **auto-tunes** kernels by timing several candidate implementations per layer and keeping the fastest; it **selects precision**, dropping to fp16 or int8 (or fp8 on newer chips) where the math allows; and it **reuses memory**, packing the network's activations into a small shared pool instead of allocating a fresh buffer per layer.

![A vertical stack showing the four jobs of the TensorRT builder: layer and tensor fusion, kernel auto-tuning, precision selection, and activation memory reuse, producing an optimized engine plan](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-2.png)

The reason these four together can produce a 2–5× speedup, rather than the 1.2× you might guess for "some kernel tweaks," is that a naive runtime leaves enormous performance on the table in *several independent ways at once*. It is memory-bound where it should be compute-bound (no fusion), it uses suboptimal kernels (no tuning), it runs everything in fp32 (no precision drop), and it thrashes the allocator (no reuse). Each fix is multiplicative with the others. Fuse and you cut memory traffic; then drop to fp16 and you halve the remaining traffic *and* double the arithmetic throughput on tensor cores; then auto-tuning makes sure you are using the tensor-core kernel at all. The wins stack.

## 2. The science: why fusion is the biggest free lunch

Start with fusion, because it is the one whose payoff is most counterintuitive and most provable. Consider the most common pattern in a CNN: a convolution, followed by adding a bias, followed by a ReLU. Three operations. A naive runtime runs them as three kernels. Here is what that actually costs in memory traffic.

Let the convolution's output tensor have $N$ elements, each stored in $B$ bytes (4 for fp32, 2 for fp16). The convolution kernel computes the output and writes all $N$ elements to global memory — that is $NB$ bytes written. The bias-add kernel then *reads* those $N$ elements back ($NB$ bytes read), adds the bias, and writes them again ($NB$ written). The ReLU kernel reads them a third time ($NB$ read), applies $\max(0, x)$, and writes them ($NB$ written). For the bias and ReLU steps alone — operations that do essentially zero arithmetic per element — you have moved $4NB$ bytes through global memory. The arithmetic is a rounding error; the whole cost is memory traffic.

This is the textbook definition of a **memory-bound** kernel: one whose runtime is dominated by data movement, not by floating-point math. The [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) makes this precise. A kernel's **arithmetic intensity** $I$ is the number of FLOPs it does per byte of memory traffic, $I = \text{FLOPs} / \text{bytes}$. The bias-add does one FLOP (an add) per element and moves $2B$ bytes per element (read + write), so $I = 1/(2B) = 0.125$ FLOP/byte in fp32. ReLU is the same. These intensities are far to the left of the roofline's ridge point — the machine could do hundreds of FLOPs per byte but these kernels ask for a fraction of one, so they run at memory bandwidth, wasting the compute units entirely.

Fusion fixes this by *not writing the intermediate to memory at all*. TensorRT compiles conv + bias + ReLU into a single kernel that computes the convolution output in registers or shared memory, adds the bias and applies the ReLU on those in-register values, and writes the final result to global memory exactly once. The $4NB$ bytes of pointless bias/ReLU traffic vanish. You read the inputs once, write the output once, and the bias and activation come along for free. The arithmetic intensity of the *fused* kernel is dominated by the convolution's real math, which is high, so the fused kernel sits much closer to the compute roof.

We can put a number on the saving. For a layer whose fused kernel is compute-bound but whose unfused version is memory-bound on the elementwise ops, the time saved is roughly the time to move those eliminated bytes:

$$
\Delta t_{\text{fusion}} \approx \frac{4NB}{\text{BW}_{\text{mem}}},
$$

where $\text{BW}_{\text{mem}}$ is the GPU's achievable global-memory bandwidth. On a Jetson Orin Nano with roughly $68\ \text{GB/s}$ of memory bandwidth, a single feature map of $N = 256 \times 56 \times 56 \approx 8.0 \times 10^5$ elements in fp16 ($B = 2$) carries $4NB \approx 6.4\ \text{MB}$ of avoidable traffic, costing about $6.4\ \text{MB} / 68\ \text{GB/s} \approx 94\ \mu\text{s}$ *per such fusion site*. A ResNet has dozens of conv-bias-activation sites; multiply and you see how fusion alone can shave milliseconds. This is why fusion is the biggest single lever in the builder: it attacks the memory wall, and on edge GPUs with modest bandwidth the memory wall is usually the wall.

TensorRT fuses much more than conv-bias-relu. It does **vertical fusion** (chains of pointwise ops collapse into their producer), **horizontal fusion** (sibling layers that share an input — like the three convolutions in an inception block — merge into one wider kernel that launches once instead of three times), and **attention fusion** (the multi-head attention pattern — the QK matmul, the scale, the softmax, the AV matmul — collapses into a single fused multi-head-attention kernel, the same idea as FlashAttention, avoiding materializing the $L \times L$ attention matrix to memory). For transformer-heavy models, attention fusion is frequently the dominant win, for exactly the same memory-traffic reason as conv fusion: the $L \times L$ scores matrix is enormous and writing it out is pure waste.

#### Worked example: counting the fusion savings on one block

Take a single MobileNet-style inverted residual block on a Jetson Orin Nano, fp16. The block is: 1×1 expand conv → batchnorm → ReLU6 → 3×3 depthwise conv → batchnorm → ReLU6 → 1×1 project conv → batchnorm, plus a residual add. That is eight ops in the graph, of which three are convolutions doing real math and five (three batchnorms, two ReLU6s) plus the residual add are elementwise and memory-bound.

Unfused, each of those five elementwise ops and the add reads and writes the full activation tensor. Say the expanded activation is $N = 96 \times 28 \times 28 \approx 7.5 \times 10^4$ elements at $B = 2$ bytes. Each elementwise op moves $2NB \approx 0.3\ \text{MB}$ (read + write); six of them is $\approx 1.8\ \text{MB}$ of traffic that does almost no arithmetic. At $68\ \text{GB/s}$ that is $\approx 26\ \mu\text{s}$ of pure memory time per block. TensorRT folds every batchnorm into the preceding convolution's weights (a batchnorm at inference is just an affine scale-and-shift, foldable into conv weights and bias for free), fuses each ReLU6 into its conv, and fuses the residual add into the project conv's epilogue. The five elementwise kernels and the add **disappear** as separate launches. Across a 17-block MobileNetV3, eliminating $\approx 26\ \mu\text{s}$ of elementwise traffic per block is on the order of $0.4\ \text{ms}$ saved, plus the launch-overhead savings of firing roughly $6 \times 17 \approx 100$ fewer kernels — at $\sim 5\ \mu\text{s}$ of CPU-side launch latency each on a Jetson, another $\approx 0.5\ \text{ms}$. On a model whose total fp16 latency is a few milliseconds, that is a large fraction, and it is *before* any precision drop.

## 3. Kernel auto-tuning: why your engine is not portable

The second builder job is the one that makes TensorRT a *measurement* tool, not just a graph rewriter, and it is the direct cause of engine non-portability. For any given layer — say a 3×3 convolution with a specific input shape, stride, and channel count — there is not one CUDA kernel that implements it. There are many. There is an implicit-GEMM kernel, a Winograd kernel, an FFT-based kernel, several direct-convolution kernels with different tiling and different use of tensor cores, and within each there are tunable parameters (tile sizes, how work maps to thread blocks). Which one is fastest depends intricately on the GPU's number of streaming multiprocessors, its tensor-core generation, its cache sizes, its memory bandwidth, and the exact tensor dimensions. There is no way to know the winner by reasoning. You have to *time* them.

That is exactly what the builder does. For each layer it enumerates the candidate kernels — TensorRT calls each candidate a **tactic** — and benchmarks them on the actual target GPU, then keeps the fastest one and records its choice in the engine. Figure 3 contrasts this with a naive runtime: the naive path calls one generic, untuned kernel per layer with no timing, while the builder times several tactics on the real chip and bakes only the winner into the plan.

![A two column before and after diagram contrasting a naive runtime that calls one generic untuned kernel per layer with the TensorRT builder that times several kernel tactics on the target GPU and bakes only the fastest into the engine](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-3.png)

Two consequences fall out immediately. First, this is *why the build is slow*: timing dozens of tactics across dozens of layers, each run multiple times for a stable measurement, takes minutes to tens of minutes. It is genuine benchmarking. Second, this is *why the engine is non-portable*: the winning tactic for a 3×3 conv on an Orin Nano's Ampere GPU with 1024 CUDA cores is very often **not** the winner on a desktop RTX 4090's Ada GPU with 16,384 cores, because the optimal tiling and tensor-core usage differ. The engine has baked in choices that are correct only for the chip they were measured on. Ship that engine elsewhere and you would be running suboptimal kernels even if it loaded — which, as we will see in the portability section, it usually will not.

The amount of timing the builder does is controlled by the **builder optimization level** (`--builderOptimizationLevel`, default 3, range 0–5 in recent TensorRT) and the available **workspace** memory it is allowed to use for scratch during tactic timing. Higher optimization levels try more tactics and produce faster engines at the cost of longer builds; level 5 can take several times longer than level 3 for a single-digit-percent latency gain. A larger workspace lets the builder consider memory-hungry tactics (like Winograd or large-tile GEMMs) that might be faster but need scratch space — starve the workspace and you forbid those tactics, so on a memory-constrained Jetson it is worth setting a generous but realistic `--memPoolSize=workspace:N` rather than letting it default low.

There is a real engineering lesson buried here. Because the builder times kernels on the actual device, **you should build on the device you will deploy on, or on an identical one.** Building a Jetson engine on a desktop GPU and copying it over is one of the most common TensorRT mistakes, and it fails for two reasons at once: the tactics are wrong for the Jetson, and (usually) the engine will not even deserialize because the GPU architecture differs. The supported, robust workflow is to build *on the Jetson* (or in a container matched to the Jetson's TensorRT and CUDA versions), accepting the slower on-device build as the price of a correctly tuned engine. For CI you keep a Jetson in the loop or use NVIDIA's cross-compilation tooling that targets a specific SM and TensorRT version explicitly.

A subtle point that trips people up: the builder is *stochastic* in a small way. Kernel timings on a real GPU have run-to-run jitter from thermal state, clock variation, and scheduler noise, so two builds of the same ONNX on the same chip can occasionally pick different tactics for a borderline layer and end up a percent or two apart in latency. This is normal and expected — it is the price of measurement-based optimization. If you need byte-for-byte reproducible engines (for a signed artifact, say), pin the clocks with `jetson_clocks` before building so the timings are stable, set a fixed builder optimization level, and accept that even then the guarantee is "very close," not "identical." Treat the engine as a measured artifact with a small tolerance, not a deterministic compile output.

### The fourth job: activation memory reuse

The builder's fourth job gets the least attention and matters most on memory-starved edge devices, so it deserves its own treatment. A naive runtime allocates a fresh buffer for every layer's output and frees it whenever the allocator gets around to it, so peak memory is roughly the sum of all activation tensors that are simultaneously live — which on a deep network can be large. TensorRT instead computes a **memory plan** at build time: it analyzes the lifetime of every activation tensor (the span from when a layer produces it to the last layer that consumes it) and packs non-overlapping lifetimes into the *same* physical memory. Two tensors that are never alive at the same moment share one buffer.

This is exactly the classic register-allocation problem from compilers, applied to GPU memory. Formally, you have a set of tensors each with a live interval $[t_{\text{produce}}, t_{\text{last-consume}}]$ and a size in bytes, and you want to assign each to an offset in a single pool so that any two tensors with overlapping intervals get disjoint byte ranges, minimizing the pool's total size. It is an interval-graph coloring problem, and TensorRT solves it greedily at build time. The payoff is that the engine's activation memory — what TensorRT calls the **device memory** of the execution context — is far smaller than the naive sum-of-activations, often by 2–4×, because most activations in a feed-forward network are short-lived and can overlap a single residual or skip tensor that lives long.

Why this matters specifically on edge: a Jetson Orin Nano has 8 GB of memory *shared* between CPU and GPU (unified memory), and that budget holds the OS, your application, the engine's weights, and the activation pool all at once. On a desktop with 24 GB of dedicated VRAM you rarely think about activation memory; on a Jetson it can be the difference between the engine loading and an out-of-memory crash, especially with large input resolutions or large `max` shapes in an optimization profile. The memory plan is also why you should keep `max` shapes tight — the activation pool is sized for the worst case in the profile, so a needlessly large `max` reserves memory you never use. When a Jetson engine OOMs, the first two things to check are the workspace size (build-time scratch, freed after build) and the activation pool (runtime, sized by `max` shape); they are different memory at different times and people conflate them.

## 4. Precision selection and the INT8 calibration problem

The third builder job is precision, and this is where TensorRT touches the rest of this series most directly. By default the builder works in fp32. Pass `--fp16` and you *permit* it to run layers in half precision wherever that is faster and the builder judges it numerically acceptable; pass `--int8` and you permit 8-bit integer arithmetic; on Hopper and newer, `--fp8` permits 8-bit float. These flags are permissions, not commands — the builder still chooses per layer, and it can keep a sensitive layer in higher precision if mixing wins. That per-layer freedom is the same idea as [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) in the quantization track: not every layer wants the same number of bits.

fp16 is nearly free. The format keeps a 10-bit mantissa and 5-bit exponent, and for the dynamic range of typical activations and weights it loses almost no accuracy while halving memory traffic and doubling tensor-core throughput. In my opening example, fp16 took latency from 12.6 ms to 6.1 ms with *zero* measurable accuracy change. The standard advice is correct: **always try fp16 first.** It is the cheapest 2× you will ever get on an NVIDIA GPU, it almost never moves accuracy, and it requires no calibration data.

There is one fp16 caveat worth knowing because it does occasionally bite: fp16's *exponent* is only 5 bits, giving a maximum representable magnitude around 65504. A layer that produces very large intermediate values — an unnormalized attention score, a sum over a long reduction, an exponential — can overflow to `inf` in fp16 and then propagate `NaN` through the rest of the network, silently wrecking the output. This is rare in well-normalized vision models (batchnorm keeps activations in a sane range) and more common in transformers and in custom architectures with unbounded ops. The fixes are exactly what the builder's per-layer freedom enables: TensorRT will keep an overflow-prone layer in fp32 if you mark it, or you use `bf16` where supported (same 8-bit exponent as fp32, so it cannot overflow where fp32 would not), or you add normalization upstream. The reason `bf16` exists at all is this overflow problem — it trades fp16's mantissa precision for fp32's dynamic range, which is the right trade for training and for numerically wide inference. On inference for most edge CNNs, plain fp16 is fine; just know that an `inf`/`NaN` after an fp16 build points at a dynamic-range overflow, not a bug in TensorRT.

The per-layer precision freedom is worth dwelling on because it is the bridge to [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis). When you pass `--int8 --fp16`, you are not saying "make everything int8" — you are handing the builder a menu of precisions per layer and letting it pick the fastest one that is numerically acceptable for that layer. Some layers are *sensitive*: the first convolution that sees raw pixels, the final classifier, layers with very skewed activation distributions. Quantizing those to int8 can cost disproportionate accuracy for little speed, so the right move is to keep them in fp16 (or fp32) while the heavy middle of the network runs int8. TensorRT does some of this automatically when you enable mixed precision, and you can force it with the layer-precision API (`layer.precision = trt.float16` plus `config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)`). The sensitivity-analysis recipe — quantize layers one at a time, measure the accuracy hit of each, and keep the few most-damaging layers in higher precision — recovers most of the accuracy that uniform int8 loses, at a tiny speed cost, and it is the standard tool when plain int8 misses your budget by a point or two but full QAT feels like overkill.

INT8 is where it gets interesting, because int8 has only 256 levels and a fixed, narrow representable range, so the builder must know the *scale* — the mapping from float values to those 256 integer levels — for every tensor it quantizes. For weights this is easy: the weights are fixed numbers, so the builder reads their range directly. For **activations** it is the hard problem, because an activation tensor does not exist until you run data through the network, and its range depends on the input. To pick a good scale, TensorRT needs to observe the activations on representative data. That observation process is **calibration**, and it is the same range-estimation problem covered in depth in [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) — here it is wired into the TensorRT builder specifically.

Figure 4 shows the calibration flow as a sequence: feed a few hundred representative inputs, collect per-tensor activation histograms as the data streams through, run a calibrator algorithm to turn each histogram into a scale, and write a **calibration cache** so that the next build can skip the entire data-streaming pass and reuse the scales.

![A left to right timeline showing INT8 calibration: representative data in, per-tensor histograms collected, an entropy calibrator deriving scales, per-tensor dynamic ranges set, and a calibration cache written for reuse](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-4.png)

TensorRT ships two main calibrator algorithms. **Entropy calibration** (the default, `IInt8EntropyCalibrator2`) builds a histogram of each activation tensor's absolute values and then searches for the clipping threshold that minimizes the Kullback-Leibler divergence between the original fp32 distribution and the int8-quantized one. Intuitively, it picks the range that loses the least *information*, which usually means clipping the outlier tail so the bulk of the distribution gets a fine step size — exactly the clipping-versus-quantization-error trade-off the PTQ post derives. **MinMax calibration** (`IInt8MinMaxCalibrator`) just uses the observed min and max, which is simpler and sometimes better for distributions with meaningful tails (it is the common choice for some transformer activations and for detector heads where the extremes carry signal).

The science of why this matters: int8 quantization replaces a real value $x$ with $\hat{x} = s \cdot \mathrm{round}(x/s)$, clipped to the representable range, where $s$ is the scale (TensorRT uses symmetric per-tensor quantization for activations and supports per-channel for weights). The rounding introduces error well modeled as uniform noise on $[-s/2, s/2]$ with variance $\sigma_q^2 = s^2/12$. The scale is set by the chosen range: $s = \beta / 127$ for a symmetric quantizer with clip magnitude $\beta$. So the quantization noise grows with $\beta^2$ — pick the range too wide to admit one outlier and you coarsen every ordinary value quadratically. The signal-to-quantization-noise ratio for a well-matched range follows the familiar law

$$
\text{SQNR} \approx 6.02\,b + 1.76\ \text{dB},
$$

about 6 dB — one bit of effective precision — per actual bit. But that law *assumes the range matches the signal*. A loose range from one outlier throws away effective bits: if $\beta$ is 8× larger than the bulk needs, you have wasted three bits and your int8 quantizer behaves like int5 on the values that matter. **This is exactly what entropy calibration is preventing** — it is choosing $\beta$ to keep as many effective bits as possible, by trading a little clipping error for a lot less quantization error. A bad calibration set, or the wrong calibrator, does not lose accuracy in some vague way; it provably costs you effective bits on every activation.

The calibration cache deserves emphasis because it is a real workflow win. Once you have calibrated, TensorRT writes the per-tensor scales to a small text file. On the next build — say you are rebuilding for a new TensorRT version, or sweeping builder options — you point the builder at that cache and it skips the entire calibration data pass, reading scales straight from the file. This turns a calibration that might take a minute of forward passes into an instant lookup, and it lets you check the scales into version control so a teammate's build is byte-for-byte reproducible without shipping the calibration dataset. The one caveat: a cache is only valid for the same network and the same calibrator; change the model or the calibrator algorithm and you must recalibrate.

### When to use QAT instead of post-training int8

If post-training int8 calibration loses too much accuracy — more than your budget allows after you have tried entropy versus minmax and a better calibration set — the answer is **quantization-aware training**, which lets the model *learn* to be robust to the rounding during a fine-tuning run. TensorRT consumes QAT models through ONNX's `QuantizeLinear`/`DequantizeLinear` (Q/DQ) nodes: you insert fake-quant nodes in PyTorch (via `pytorch-quantization` or `torch.ao.quantization`), fine-tune, export to ONNX with the Q/DQ nodes embedded, and the builder reads the scales from those nodes instead of calibrating. This is the **explicit-quantization** path, and it gives you per-layer control and reproducibility at the cost of a training run. The [QAT post](/blog/machine-learning/edge-ai/quantization-aware-training-qat) covers the fake-quant mechanics; the point here is that TensorRT supports both the post-training (implicit, calibrated) and QAT (explicit, Q/DQ) routes, and you graduate from the first to the second only when the free path runs out.

## 5. Building an engine with trtexec

Enough theory; let us build something. `trtexec` is the command-line tool that ships with TensorRT, and it is the fastest way to build an engine and profile it. It will not replace the Python API for production (you cannot wire a custom calibrator's data loader into it as flexibly), but for "how fast does this ONNX run on this chip at this precision," nothing beats it.

Here is the canonical sequence on a Jetson, building from an exported ONNX file. First, the fp32 baseline so you have a reference:

```bash
# fp32 baseline engine + profile
trtexec \
  --onnx=resnet50.onnx \
  --saveEngine=resnet50_fp32.plan \
  --memPoolSize=workspace:2048 \
  --iterations=200 --warmUp=500 --avgRuns=100 \
  --noDataTransfers --useCudaGraph
```

The flags matter. `--saveEngine` serializes the built engine to a `.plan` file you can load later without rebuilding. `--memPoolSize=workspace:2048` gives the builder 2 GB of scratch for tactic timing — on a memory-tight Jetson, tune this down if you must, but starving it forbids the fastest tactics. `--warmUp=500` runs 500 ms of inference before timing, which is *essential* on a Jetson because the GPU clocks ramp and the first kernels JIT — measuring cold gives you garbage. `--iterations` and `--avgRuns` control how many timed runs are averaged. `--noDataTransfers` times pure GPU compute by skipping host-device copies (use this to isolate the engine; drop it to measure end-to-end including PCIe/unified-memory transfers). `--useCudaGraph` captures the launch sequence into a CUDA graph to remove per-launch CPU overhead, which is significant at batch 1 on a Jetson where the CPU is weak.

Now the fp16 engine — almost always your real baseline:

```bash
# fp16 engine
trtexec \
  --onnx=resnet50.onnx \
  --fp16 \
  --saveEngine=resnet50_fp16.plan \
  --memPoolSize=workspace:2048 \
  --warmUp=500 --avgRuns=100 --useCudaGraph
```

And the int8 engine, with calibration. `trtexec` can calibrate from a directory of raw input tensors and write a cache:

```bash
# int8 engine with calibration, writing a cache
trtexec \
  --onnx=resnet50.onnx \
  --int8 --fp16 \
  --calib=resnet50_calib.cache \
  --saveEngine=resnet50_int8.plan \
  --memPoolSize=workspace:2048 \
  --warmUp=500 --avgRuns=100 --useCudaGraph
```

Note `--int8 --fp16` together: this enables **mixed int8/fp16**, letting the builder keep a layer in fp16 if int8 would be both slower and less accurate for it — almost always what you want, because forcing pure int8 on a layer that does not benefit just costs accuracy for no speed. The `--calib` flag points at the cache; if the cache does not exist and you have not supplied calibration data, `trtexec` will warn and fall back to a naive range, which produces a buildable but inaccurate engine — a classic gotcha where the engine "works" but accuracy is silently wrecked. For a real int8 build you supply calibration through the Python API (next section) and reuse the cache here.

A few more `trtexec` flags you will reach for constantly. `--dumpProfile` prints per-layer timing so you can see which layers dominate. `--exportProfile=prof.json` and `--exportLayerInfo=layers.json` dump machine-readable timing and the fused-layer structure — invaluable for confirming that fusion actually happened. `--best` is shorthand for "enable every precision and let the builder pick," useful for a quick ceiling. `--useDLACore=0 --allowGPUFallback` routes layers to the Jetson's DLA (more on that below). And `--verbose` shows you every tactic the builder considered, which is how you debug a layer that refused to fuse or fell back to fp32.

#### Worked example: the precision sweep on Jetson Orin

Here is the full sweep on a Jetson Orin Nano (8 GB), ResNet-50, ImageNet, batch 1, measured with warm-up and CUDA graphs, compared against the ONNX-Runtime-CUDA baseline that started this post. These are representative measured numbers; treat exact figures as approximate but the *shape* is what holds in practice.

| Runtime / precision | p50 latency | Throughput | Top-1 | Power (avg) |
|---|---|---|---|---|
| ONNX Runtime CUDA (fp32) | 12.6 ms | 79 img/s | 76.1% | 14.0 W |
| TensorRT fp32 | 9.8 ms | 102 img/s | 76.1% | 13.6 W |
| TensorRT fp16 | 6.1 ms | 164 img/s | 76.1% | 11.5 W |
| TensorRT int8 (entropy) | 3.4 ms | 294 img/s | 75.6% | 9.2 W |

Read this carefully because every row teaches something. TensorRT fp32 alone beats ONNX-Runtime-fp32 by ~1.3× — that is pure fusion plus auto-tuning, *no precision change*, which proves the speedup is not "just quantization." fp16 doubles throughput over fp32 for free with zero accuracy loss — the always-try-this row. int8 nearly doubles again over fp16, taking total speedup to 3.7× over the original baseline, at a cost of 0.5 top-1 points, and — crucially — it also *drops power* from 14 W to 9.2 W, because doing the same work in fewer, cheaper integer operations burns less energy. On a battery-powered or thermally-limited edge device, that power column is often the real reason to go int8, not the latency. This is the [accuracy–efficiency Pareto frontier](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) in one table: each step trades a little accuracy for a lot of latency, throughput, and watts.

Figure 6 lays the precision trade out as a matrix so the pattern is unmissable: fp16 roughly halves fp32 latency for free, int8 nearly halves it again with a small accuracy cost, and the power column falls the whole way down. The decision rule the matrix encodes is simple — pick the lowest precision that still holds your accuracy budget, because every step down buys latency, throughput, and watts together.

![A three by three matrix comparing fp32, fp16, and int8 on Jetson Orin across latency, throughput, and combined top-1 accuracy and power, showing fp16 halves fp32 latency for free and int8 nearly halves it again at a small accuracy cost](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-6.png)

One subtlety the table hides and the matrix makes you confront: the accuracy number that matters is *measured on a held-out validation set*, never on the calibration set. Calibration sees a few hundred images and tunes scales to them; reporting accuracy on those same images flatters int8 because you measured on the data you fit to. The honest protocol is to calibrate on one slice of data and measure top-1 on a disjoint validation slice, exactly as you would never report training accuracy as test accuracy. I have watched a team ship an int8 engine that looked lossless on the calibration images and lost two points on real traffic, purely because the calibration slice happened to be cleaner than production. Measure on held-out data, every time.

## 6. The Python builder API

`trtexec` is great for measurement, but production wiring — a custom calibrator with your real data loader, conditional precision per layer, serializing the engine into your serving binary — wants the Python (or C++) builder API. Here is the shape of it. First, building an fp16 engine from ONNX:

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# explicit batch network (the only mode in modern TensorRT)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)
with open("resnet50.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
config.set_flag(trt.BuilderFlag.FP16)

serialized = builder.build_serialized_network(network, config)
with open("resnet50_fp16.plan", "wb") as f:
    f.write(serialized)
print("engine built and serialized")
```

The structure is always the same: a `Builder`, a `network` parsed from ONNX, a `config` that holds the flags and workspace, and `build_serialized_network` that does the slow compile and hands back bytes you write to disk. To go int8 you add a calibrator to the config. The calibrator is an object that feeds the builder batches of real data; you implement `get_batch` to hand over the next batch and the read/write methods for the cache:

```python
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data, batch_size, cache_path="calib.cache"):
        super().__init__()
        self.data = data            # np.float32 array [M, C, H, W], preprocessed
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.index = 0
        self.device_input = cuda.mem_alloc(
            batch_size * data[0].nbytes
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.data):
            return None                       # signals end of calibration
        batch = np.ascontiguousarray(
            self.data[self.index:self.index + self.batch_size]
        )
        cuda.memcpy_htod(self.device_input, batch)
        self.index += self.batch_size
        return [int(self.device_input)]       # device pointers, in input order

    def read_calibration_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                return f.read()               # reuse scales, skip the data pass
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)
```

The two cache methods are the win from the previous section: if `read_calibration_cache` returns bytes, the builder uses those scales and never calls `get_batch`, making rebuilds instant. Wiring it into the config is two lines added to the fp16 build:

```python
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)   # allow int8/fp16 mixing
calib_data = load_preprocessed_calib_images(n=512)   # your loader
config.int8_calibrator = ImageCalibrator(calib_data, batch_size=8)
```

Use a few hundred to ~1000 calibration images drawn from the *real* input distribution — the same preprocessing (resize, normalize, channel order) your production pipeline uses. The single most common int8 accuracy bug is a calibration set whose preprocessing does not match inference: if you calibrate on differently-normalized images, the scales are wrong and accuracy tanks for a reason that has nothing to do with int8 itself. A few hundred samples is plenty; calibration estimates ranges, not a full distribution, and the estimate stops moving quickly. Tiny calibration sets (a dozen images) are the stress-test failure mode — the histograms are noisy, the entropy threshold is unstable, and you get a fragile engine whose accuracy swings with which dozen images you happened to pick.

Loading and running a serialized engine is the runtime half, and it is deliberately lightweight — no builder, no compilation, just deserialize and execute:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("resnet50_int8.plan", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# allocate device buffers from the engine's I/O tensor shapes
inp_name = engine.get_tensor_name(0)
out_name = engine.get_tensor_name(1)
inp_shape = engine.get_tensor_shape(inp_name)
out_shape = engine.get_tensor_shape(out_name)

d_in = cuda.mem_alloc(int(np.prod(inp_shape)) * 4)
d_out = cuda.mem_alloc(int(np.prod(out_shape)) * 4)
context.set_tensor_address(inp_name, int(d_in))
context.set_tensor_address(out_name, int(d_out))

stream = cuda.Stream()
host_in = np.random.rand(*inp_shape).astype(np.float32)
cuda.memcpy_htod_async(d_in, np.ascontiguousarray(host_in), stream)
context.execute_async_v3(stream.handle)     # the actual inference
host_out = np.empty(out_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(host_out, d_out, stream)
stream.synchronize()
print("logits:", host_out[:5])
```

`deserialize_cuda_engine` is the step that fails loudly if the engine was built for a different GPU or TensorRT version — which is the entire portability story, and the next section. `create_execution_context` allocates the engine's working memory; you can create several contexts from one engine to run concurrent inferences sharing the same weights. `execute_async_v3` is the modern entry point that takes a stream and uses the tensor addresses you set. In production you would use a polished wrapper (`torch2trt`, NVIDIA's `Torch-TensorRT`, or the Triton TensorRT backend) rather than raw pycuda, but this is what they call underneath.

## 7. Dynamic shapes and optimization profiles

The engines above assume a fixed input shape. Real serving often does not — you want one engine that handles batch 1 for an interactive request and batch 16 for a backlog flush, or variable sequence lengths for a text model, or variable image resolutions. TensorRT handles this with **optimization profiles**: you declare, per input dimension that varies, a `min`, an `opt`, and a `max` shape, and the builder produces *one* engine that accepts any shape in that range, with kernels auto-tuned specifically for the `opt` shape.

Figure 5 contrasts the two worlds. A fixed-shape engine must be rebuilt for every distinct input size — N sizes means N engines, N builds, N artifacts to manage. An optimization profile collapses that to a single engine spanning `min` to `max`, tuned at `opt`.

![A two column before and after diagram contrasting a fixed shape engine that needs a full rebuild per input size with an optimization profile declaring min, opt, and max so one engine serves a whole range of batch sizes tuned at the opt point](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-5.png)

The build-time trade is the whole point and worth stating precisely. The builder auto-tunes kernels for the `opt` shape, so inference *at* `opt` is as fast as a fixed-shape engine built for that exact size. Inference at shapes *far* from `opt` — say `max` when `opt` is small — may use kernels that are not perfectly tuned for that size and run a little slower than a dedicated engine would. There is also a memory cost: the engine sizes its activation pool for the `max` shape, so a profile with a huge `max` reserves more device memory even when you run small. The art is picking `opt` to be your most common shape and keeping `max` no larger than you truly need.

With `trtexec`, profiles are flags:

```bash
# one engine spanning batch 1..32, tuned for batch 8, dynamic batch dim
trtexec \
  --onnx=resnet50_dynamic.onnx \
  --fp16 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:32x3x224x224 \
  --saveEngine=resnet50_dyn_fp16.plan \
  --warmUp=500 --avgRuns=100
```

This requires the ONNX model to have a *dynamic* batch dimension (exported with `dynamic_axes` in `torch.onnx.export`); a model exported with a fixed batch dimension cannot be made dynamic by `trtexec`. In the Python API you add a profile to the config:

```python
profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 3, 224, 224),
    opt=(8, 3, 224, 224),
    max=(32, 3, 224, 224),
)
config.add_optimization_profile(profile)
```

and at runtime you tell the context the actual shape for this inference before executing:

```python
context.set_input_shape("input", (4, 3, 224, 224))   # any shape in [min, max]
assert context.all_binding_shapes_specified
# ... set tensor addresses sized for this shape, then execute_async_v3
```

#### Worked example: the dynamic-shape build-time-versus-runtime trade

Suppose your service is 80% single-image interactive requests (batch 1) and 20% batched backlog flushes (batch up to 32). Three designs:

**Design A — two fixed engines.** Build `batch1.plan` (tuned for 1) and `batch32.plan` (tuned for 32). Best-case latency at each size, but two builds (say 8 min each on the Jetson), two artifacts, and runtime logic to pick the engine and reload context memory when switching. Batch 1 at p50 6.1 ms, batch 32 at p50 78 ms (≈ 2.4 ms/img amortized).

**Design B — one dynamic engine, opt=1.** One build, one artifact. Batch 1 is fully tuned: p50 6.1 ms. Batch 32 runs on kernels tuned for batch 1, so it is slower than a dedicated batch-32 engine — say p50 96 ms (≈ 3.0 ms/img), a ~23% throughput hit on the rare large batches. Activation pool sized for batch 32, so it reserves the larger memory footprint even for the common batch-1 path.

**Design C — one dynamic engine, opt=8.** One build, one artifact. Batch 1 runs on kernels tuned for 8 — slightly slower than a dedicated batch-1 engine, say p50 6.8 ms (an 11% hit on your *most common* path). Batch 32 is closer to tuned, say p50 84 ms.

The decision is a values call, and naming it is the job. If interactive p50 is your SLA and batch flushes are background, **Design B** wins: you protect the 80% path's latency exactly and eat the throughput hit on the 20% that nobody is waiting on. If you are throughput-bound on the flushes and the interactive path has slack, Design C's balance is better. Design A is only worth its operational complexity if *both* sizes are latency-critical and the 11–23% gaps are unacceptable. Nine times out of ten on an edge device I ship Design B and move on, because the simplicity of one artifact and the protected interactive latency are worth more than reclaiming milliseconds on background work. The general rule: **set `opt` to the shape whose latency you are graded on**, not the average shape.

## 8. The DLA: Jetson's other accelerator

Jetson Orin modules have a second inference engine besides the GPU: the **Deep Learning Accelerator (DLA)**, a fixed-function NPU built for convolutional inference at very low power. There are two DLA cores on Orin. The DLA is not a general GPU — it supports a restricted set of layers (mostly convolutions, pooling, activations, elementwise ops common in CNNs) — but for the layers it does support it runs them at a fraction of the GPU's power, and critically it runs *in parallel with the GPU*. The strategic use is to offload the heavy convolutional backbone to the DLA so the GPU is free for other work (preprocessing, a second model, the unsupported layers), increasing total system throughput per watt.

TensorRT targets the DLA directly. You tell the builder to put layers on a DLA core and to allow GPU fallback for layers the DLA cannot handle:

```bash
trtexec \
  --onnx=resnet50.onnx \
  --int8 --fp16 \
  --useDLACore=0 \
  --allowGPUFallback \
  --saveEngine=resnet50_dla.plan \
  --warmUp=500 --avgRuns=100
```

The honest trade: the DLA is **slower per inference than the GPU** for a single stream, but **much lower power** and it offloads the GPU. So you do not move to the DLA to make one model faster — you move to it to run a model at low power, or to run *two* models at once (one on DLA, one on GPU) for higher aggregate throughput. The stress-test gotcha is `--allowGPUFallback`: every layer the DLA does not support bounces back to the GPU, and each bounce is a synchronization point and a memory round-trip. A model with many unsupported layers scattered through it will *thrash* between DLA and GPU and end up slower than pure GPU, defeating the purpose. Before committing to the DLA, dump the layer placement (`--exportLayerInfo`) and confirm the backbone runs as a contiguous DLA block, not a Swiss cheese of fallbacks. If half your layers fall back, the DLA is the wrong call for that model.

The DLA also constrains precision: it runs fp16 and int8 (with calibration) but not fp32, and its int8 path has its own quirks, so always validate accuracy on the DLA engine specifically rather than assuming it matches the GPU int8 engine. For the official op-support matrix and the current restrictions, NVIDIA's DLA documentation is the source of truth and changes with each JetPack release.

## 9. Profiling honestly on a Jetson

A speedup you cannot measure reliably is a speedup you cannot defend. Profiling on a Jetson has device-specific traps that will give you wrong numbers if you ignore them, so this section is about measuring honestly. The principles are the same as the series' [metrics post](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device): batch-1 reality, p50 *and* p99, warm-up, thermal state.

**Lock the clocks and power mode first.** Jetsons ship with dynamic clocking and several power modes that cap the GPU and CPU frequencies and the number of active cores. Measure in an undefined power state and your numbers are noise. Before benchmarking:

```bash
# show available power modes, then set max-performance mode
sudo nvpmodel -q                 # query current mode
sudo nvpmodel -m 0               # mode 0 is MAXN (max power/clocks) on most Orin
sudo jetson_clocks               # pin GPU/CPU/EMC clocks to max, disable DVFS
sudo jetson_clocks --show        # confirm the clocks are pinned
```

`nvpmodel -m 0` selects the maximum-power profile (the mode numbers differ per module — query first). `jetson_clocks` then pins every clock to its maximum and disables the dynamic frequency scaling that otherwise makes back-to-back runs disagree. **Report which power mode you measured in** — a latency number without a power mode is meaningless, because mode 0 (MAXN, e.g. 15 W) and a 7 W mode can differ by 2× on the same engine. For a production figure you measure in the power mode you will actually *deploy* in, which is often *not* MAXN because of thermal or battery limits. State it.

**Watch power, temperature, and throttling while you run.** `tegrastats` is the Jetson's live telemetry:

```bash
tegrastats --interval 100        # GPU%, RAM, temps, and power rails every 100 ms
```

It prints GPU utilization, memory use, the temperatures of each thermal zone, and the instantaneous power draw on the GPU/CPU/SOC rails. Two things to look for: is `GR3D_FREQ` (the GPU) actually near 100% during the timed run (if not, you are bottlenecked on the CPU, on data transfers, or on launch overhead, not on the GPU compute), and is the temperature climbing toward the throttle point? On a small Jetson under sustained load the SoC heats up and the hardware *throttles* the clocks to stay in thermal budget, so a 60-second sustained benchmark can report meaningfully higher latency than a 2-second burst. **For an honest sustained number, run long enough to reach thermal steady state**, then measure — a burst benchmark flatters you.

**Use Nsight Systems for the real timeline.** When `tegrastats` shows the GPU is not saturated and you need to know *why*, `nsys` captures a full CPU-GPU timeline:

```bash
nsys profile -t cuda,nvtx,osrt \
  --output=trt_profile \
  python run_engine.py
```

Open the resulting `.nsys-rep` in the Nsight Systems UI and you see every kernel launch, every memory copy, and every gap. The classic finding on a Jetson at batch 1 is *gaps between kernels* — the GPU finishes a kernel and sits idle waiting for the weak CPU to launch the next one. That is the launch-overhead problem, and it is exactly why `--useCudaGraph` (which captures the whole launch sequence so the CPU issues it as one unit) often buys more at batch 1 on a Jetson than any kernel optimization. If the timeline is wall-to-wall kernels with no gaps, you are compute-bound and the engine is doing its job; if it is mostly gaps, fix the launch path before touching the model.

A practical measurement protocol that I trust: pin clocks with `nvpmodel`/`jetson_clocks`, warm up for 500 ms, run 200+ timed iterations, report **p50 and p99** (the p99 catches thermal/scheduler hiccups that the median hides), watch `tegrastats` to confirm the GPU is saturated and the temperature is at steady state, and state the power mode. Do all of that with `trtexec --warmUp=500 --iterations=200 --useCudaGraph` and you get numbers you can put in a design doc and defend in review.

## 10. Engine portability: the rule that bites everyone

Now the rule that catches every team eventually, stated as plainly as I can: **a TensorRT engine is locked to the GPU architecture, the TensorRT version, and (effectively) the CUDA version it was built with.** Build on machine A, the engine runs on machine A. Copy it to machine B and the runtime's `deserialize_cuda_engine` will, in the common case, *refuse to load it* — and in the less common case where it loads, it runs kernels tuned for the wrong chip.

Figure 6 shows the failure and the fix. Build an engine on an RTX 4090 with TensorRT 10.0, copy the `.plan` to a Jetson Orin Nano, and the deserialize fails with a version/architecture mismatch. The fix is not a flag — it is to *rebuild on the target*, on the Orin, with the Orin's matching TensorRT version, so the tactics are tuned for the Orin's streaming multiprocessors and the engine deserializes cleanly.

![A two column before and after diagram showing an engine built on an RTX 4090 with one TensorRT version failing to deserialize on a Jetson Orin, versus rebuilding on the Orin with a matching version so the engine runs correctly](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-7.png)

Why is this so strict? Recall what the engine *is*: a serialized record of "use this kernel tactic for this layer, in this memory layout." The kernel tactics are GPU-architecture-specific machine choices (an Ampere SM and an Ada SM have different optimal kernels), the serialization format changes between TensorRT versions, and the kernels link against a specific CUDA runtime. TensorRT deliberately refuses to load an engine across these boundaries because running the wrong kernels would be silently incorrect or crash. There is a **version-compatibility** mode (`BuilderFlag.VERSION_COMPATIBLE`) that lets an engine load across some TensorRT minor versions by embedding a lean runtime, and **hardware-compatibility** modes that target a family of GPUs at a performance cost, but these are escape hatches with caveats — the default and the assumption you should plan around is: *one engine per (GPU architecture, TensorRT version) target, rebuilt on or for that target.*

The operational consequences are real and you must design for them. Your build artifact is not "the model" — it is "the model, for this exact device, this TensorRT version." When NVIDIA ships a new JetPack with a new TensorRT, your engines are stale and you rebuild. When you add a new Jetson SKU to your fleet, you build a new engine for it. The robust pattern is to **store the ONNX (portable) as the source of truth and treat engines as a build-cache keyed by (device, TRT version)**: a fleet of Orin Nanos and Orin NXs running JetPack 6 gets two engine variants built in CI containers matched to each, the calibration cache shared across them so int8 scales are reproducible. Ship the wrong engine and the device fails to start the model — loud, at least, not silent. The silent failure mode is copying an engine that *happens* to load (same arch, compatible version) but was tuned for a different SM count, quietly leaving 20% of your speedup on the floor; catch that by validating measured latency on each target, not by trusting that "it loaded."

## 11. Where TensorRT fits — and where it does not

TensorRT is not free, and a principal engineer's job is to name the cost. Let me be opinionated about when to reach for it and when to walk away. Figure 8 is the decision as a tree: start from "need fast inference on the edge," branch on whether the target is NVIDIA, whether the fleet is portable/multi-vendor, and whether the workload is an LLM, and land on a concrete tool.

![A decision tree for choosing TensorRT, branching from needing fast edge inference through whether the target is NVIDIA, a portable multi-vendor fleet, or an LLM, landing on building a TensorRT engine, using ONNX Runtime, or using TensorRT-LLM](/imgs/blogs/tensorrt-and-gpu-edge-inference-on-jetson-8.png)

**Reach for TensorRT when:** your target is NVIDIA (Jetson or an NVIDIA GPU server), latency or throughput-per-watt is the binding constraint, and you control the deployment tightly enough to rebuild engines per target. This is its sweet spot and it is unmatched there — nothing else hits an NVIDIA chip's peak the way a TensorRT engine does. If you are shipping a vision model to a Jetson product line and the latency SLA is real, TensorRT is not optional; it is the default, and fp16 then int8 is the path.

**Walk away from TensorRT when:** your fleet is multi-vendor or portable (you also ship to Qualcomm NPUs, Apple Neural Engine, Intel, AMD, or plain ARM CPUs) and you cannot afford a separate optimization stack per vendor. Then a single portable runtime — [ONNX Runtime](/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving) with the appropriate execution provider per device — buys you one model artifact across all of them at the cost of leaving some NVIDIA-specific peak on the table. The trade is *portability versus peak*: TensorRT gives peak on NVIDIA only; ONNX Runtime gives "good enough everywhere." For a heterogeneous fleet, the operational simplicity of one artifact often wins even though each NVIDIA device runs a little slower than a hand-built engine would.

The other honest cost is **build complexity**. TensorRT is a real compiler with a real learning curve: calibration data loaders, the implicit-versus-explicit quantization split, optimization profiles, DLA op support, version pinning, the engine-rebuild lifecycle. If your model is small and already fast enough through ONNX Runtime CUDA, and you have no latency or power pressure, the engineering cost of standing up a TensorRT build pipeline may exceed the benefit — "is it fast enough already?" is always the first question, and sometimes the answer is yes and you should not pull this lever. Pull it when the speedup or the watts genuinely matter, not reflexively.

**For LLMs, the answer branches to a sibling tool.** Plain TensorRT can run a transformer, but autoregressive decoding has its own demons — the KV cache, in-flight (continuous) batching, paged attention, speculative decoding — that the base TensorRT does not handle well. NVIDIA's answer is **TensorRT-LLM**, a library on top of TensorRT purpose-built for LLM serving, with paged KV cache, in-flight batching, FP8 attention, and multi-GPU sharding, typically served through Triton. If your edge workload is an LLM (a Jetson AGX Orin running a 7B model, an edge server doing RAG), reach for TensorRT-LLM, not raw TensorRT; the deep dive on [how the TensorRT inference compiler works end to end](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler) covers that full LLM path. On non-NVIDIA edge for LLMs, the comparison shifts to runtimes like `llama.cpp` and MLC-LLM covered elsewhere in this series.

To make the runtime choice concrete, here is how the main edge-inference options trade off. The columns are the decisions that actually drive the call — what hardware you can target, whether one artifact serves many devices, how peak the peak is, and the build/operational cost:

| Runtime | Hardware | Portability | Peak on NVIDIA | LLM support | Build cost |
|---|---|---|---|---|---|
| TensorRT | NVIDIA only | None (rebuild per target) | Highest | Weak (use TRT-LLM) | High (compile + calibrate) |
| TensorRT-LLM | NVIDIA only | None (rebuild per target) | Highest for LLMs | Purpose-built | High |
| ONNX Runtime | Many (EP per vendor) | One model, many HW | Good, not peak | Moderate | Low |
| llama.cpp / GGUF | CPU + many GPUs | High (one GGUF) | Below TRT-LLM | LLM-focused | Low |

The honest reading of this table: TensorRT and TensorRT-LLM win the "peak on NVIDIA" column outright and lose every portability column; ONNX Runtime wins portability and operational simplicity while conceding peak; `llama.cpp` is the portable LLM default when NVIDIA-peak is not the requirement. There is no universally best row — the right choice is whichever row matches your fleet and your binding constraint. If you have one NVIDIA target and a hard latency SLA, the top row is correct and the others are leaving performance on the floor. If you have a five-vendor fleet and a "good enough" latency target, the third row is correct and TensorRT would cost you four extra optimization stacks for a peak you do not need.

One more cost worth naming explicitly because it surprises teams: the **engine-rebuild lifecycle is forever**. Unlike a portable model file you build once and forget, TensorRT engines are stale the moment NVIDIA ships a new JetPack, you add a new device SKU, or you change the ONNX. You are signing up for a CI job that rebuilds and re-validates engines on every such event, for the life of the product. That is not a reason to avoid TensorRT — the speedup is usually worth it — but it is a real, recurring operational tax that should go in your decision honestly, not get discovered six months in when a JetPack upgrade bricks every device's engine.

## 12. Case studies and real numbers

A few results from the literature and from production that anchor the claims above. As always, treat exact figures as approximate where I flag them, and trust the *shape* of the result.

**NVIDIA's own ResNet-50 int8 numbers.** NVIDIA has long published int8 TensorRT results showing roughly a 3–4× throughput gain over fp32 with under 0.5% top-1 loss on ResNet-50 with entropy calibration — consistent with the sweep table above. The headline lesson from their calibration whitepaper (Szymon Migacz, NVIDIA, 2017, "8-bit Inference with TensorRT") is exactly the entropy-calibration result this post derives: minimizing the KL divergence between the fp32 and int8 activation distributions picks a clipping threshold that beats naive min/max and recovers most of the lost accuracy. That whitepaper is the canonical reference for *why* TensorRT's default calibrator is entropy-based.

**Jetson Orin Nano vision pipelines.** NVIDIA's Jetson benchmarks for the Orin Nano show int8 TensorRT engines for detection and classification backbones (ResNet, EfficientNet, YOLO variants) running multiples faster than fp16, with the power draw dropping into the single-digit watts — the same latency-and-watts double win as the table. The exact numbers move with JetPack/TensorRT versions, which is itself the portability lesson: NVIDIA re-publishes the benchmarks per JetPack because the engines are rebuilt per version.

**The fp16 free lunch is real across architectures.** Across CNNs and transformers, the consistent industry finding is that fp16 (or bf16) inference on tensor-core GPUs costs essentially zero accuracy while roughly doubling throughput over fp32, because the formats' dynamic range covers typical activations and the rounding error is far below the model's noise floor. This is why "always try fp16 first" is not a hedge — it is a near-universal free 2×, and TensorRT applies it automatically with one flag.

**Attention fusion on transformers.** For transformer encoders served through TensorRT, the fused multi-head-attention kernel (the FlashAttention-style fusion) is frequently the dominant speedup, for the memory-traffic reason in the fusion section: not materializing the $L \times L$ attention matrix saves $O(L^2)$ bytes of traffic per head per layer. On long sequences this is the difference between memory-bound and compute-bound attention, and it is why TensorRT (and TensorRT-LLM) invest so heavily in attention fusion specifically.

**DLA offload for multi-stream throughput.** A pattern I have shipped on a Jetson Orin: a perception stack running two models — a detection backbone and a segmentation head — where running both on the GPU serially missed the frame budget. Moving the detection backbone to one DLA core (it was convolution-heavy and ran as a contiguous DLA block with no fallback thrash) freed the GPU to run segmentation, and the two ran *concurrently* — DLA and GPU are separate hardware. Single-model latency on the DLA was worse than on the GPU, exactly as the DLA section warns, but *system* throughput went up because two models now overlapped, and the total power dropped because the DLA is far more efficient per inference than the GPU for the layers it supports. The lesson generalizes: on a Jetson, the DLA's value is parallelism and watts at the system level, not single-stream speed, and you only capture it when the offloaded model maps cleanly onto supported ops.

**The build-on-target lesson, learned the hard way.** A team I worked with built their int8 engines in a desktop CI container with a different GPU and TensorRT version than the Jetson fleet, then wondered why engines either failed to deserialize on device or — worse — loaded but ran 20% slower than a locally built engine on the same chip. Both symptoms trace to the same root: tactics tuned for the wrong SM and a serialization format from the wrong version. The fix was to move the engine build into a Jetson-matched container in CI (NVIDIA ships base images per JetPack) and validate measured p50 on a physical device before promoting an engine, treating "it deserialized" as necessary but not sufficient. After that, the silent-slowdown class of bug disappeared, because the validation step caught any engine that loaded but underperformed.

## 13. Stress tests: where this breaks

A technique is only understood once you know its failure modes. Here are the ones that have bitten me, posed as the engineering questions they answer.

**What happens at int4?** TensorRT's mainstream calibrated PTQ path targets int8; sub-8-bit on the edge GPU is not the smooth slope int8 is. The SQNR law makes this concrete — dropping from 8 to 4 bits cuts effective precision in half and quadruples quantization-noise variance for the same range, so the clipping/quantization trade has far less slack and accuracy falls off a cliff for most CNNs without QAT. For LLMs, weight-only int4 is a different story handled by TensorRT-LLM's quantization (and by GPTQ/AWQ-style methods covered in [weight-only quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)), not by base TensorRT's activation calibrator. The lesson: do not expect base TensorRT int4 to behave like int8; for sub-8-bit, reach for the LLM-specific stack or QAT.

**When the calibration set is tiny.** With a dozen calibration images the entropy histograms are noisy, the chosen clipping thresholds are unstable, and accuracy swings with which dozen you picked. The fix is more data (a few hundred to a thousand, from the real distribution) — but the deeper fix when even that is not enough is QAT, which learns robustness rather than estimating it. If you cannot get representative calibration data at all (privacy, a sensor you do not have access to), int8 PTQ is the wrong tool and you should stay in fp16.

**When a layer is unsupported and falls back.** Both the DLA fallback and, occasionally, a custom op the ONNX parser does not recognize cause a *fallback* — the layer runs on a slower path (GPU instead of DLA, or a plugin/CPU instead of a fused kernel) with a synchronization and a memory round-trip at the boundary. A model peppered with fallbacks thrashes and loses the speedup. Diagnose with `--exportLayerInfo`/`--verbose`: if the backbone is not a contiguous optimized block, either rewrite the offending op into supported primitives, or write a TensorRT plugin for it, or accept the GPU path. Never assume `--allowGPUFallback` is free; it is a correctness crutch with a performance bill.

**When the model is memory-bound, not compute-bound.** TensorRT's fusion and precision tricks attack memory traffic and arithmetic throughput, but if your model is fundamentally memory-bound — tiny batches of a model dominated by elementwise ops, or an LLM in the decode phase where each token reads the whole weight matrix once — the win comes from *reducing bytes moved*, which means precision (fp16/int8/fp8 weights) and KV-cache management, not from auto-tuning compute kernels. Knowing whether you are memory- or compute-bound (the [roofline](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) tells you) decides which lever actually moves your latency. TensorRT helps with both, but through different mechanisms, and pulling the compute lever on a memory-bound model wastes your time.

## 14. When to reach for this, decisively

Putting it together as a recommendation you can act on:

- **Always start with fp16.** It is a near-free ~2× on NVIDIA tensor cores with no calibration and no measurable accuracy loss. If fp16 hits your SLA, you are done — do not add int8 complexity for speed you do not need.
- **Go int8 when you need the latency, the throughput, or the watts** that fp16 does not give, and you can supply a few hundred representative calibration images. Use `--int8 --fp16` together so the builder can keep sensitive layers in fp16. Validate accuracy on a held-out set, not on the calibration set.
- **Graduate to QAT only when post-training int8 misses your accuracy budget** after you have tried entropy versus minmax and a better calibration set. QAT costs a training run; spend it only when the free path runs out.
- **Build on the target.** Build Jetson engines on the Jetson (or a matched container), accept the slower on-device build, and never copy engines across GPU architectures or TensorRT versions. Treat ONNX as the portable source of truth and engines as a per-target build cache.
- **Use the DLA for power and offload, not single-stream speed** — and only if the backbone runs as a contiguous DLA block without fallback thrash.
- **For LLMs, use TensorRT-LLM, not raw TensorRT.** For multi-vendor fleets, use ONNX Runtime and accept "good enough everywhere" over "peak on NVIDIA only."

## Key takeaways

1. **The builder is a compiler; the engine is its target-specific binary.** Builds are slow because they benchmark kernels; engines are non-portable because they bake in choices tuned for one GPU architecture and TensorRT version.
2. **Fusion is the biggest free lunch** because it kills memory traffic on memory-bound elementwise ops — conv+bias+relu becomes one kernel, batchnorms fold into convs, attention fuses FlashAttention-style. On bandwidth-limited edge GPUs the memory wall is usually the wall.
3. **Kernel auto-tuning times multiple tactics per layer on the real chip** and keeps the fastest — which is exactly why the engine only runs at peak on the GPU it was built for. Build on the device you deploy on.
4. **fp16 is a near-free 2× with no calibration; int8 is another ~2× at sub-one-point accuracy cost plus a real power drop.** Always try fp16 first; reach for int8 when latency or watts demand it.
5. **INT8 calibration is range estimation, and entropy calibration minimizes KL divergence** to keep effective bits — a loose range from one outlier provably throws away bits. Use a few hundred representative images with matching preprocessing, and cache the scales.
6. **Optimization profiles give one engine a min/opt/max shape range, tuned at opt.** Set `opt` to the shape whose latency you are graded on, keep `max` no bigger than needed, and accept slightly slower inference far from `opt`.
7. **Profile honestly on Jetson:** pin clocks with `nvpmodel`/`jetson_clocks`, warm up, report p50 and p99 and the power mode, watch `tegrastats` for GPU saturation and thermal throttling, and use Nsight to find launch-overhead gaps that CUDA graphs fix.
8. **Engines are locked to (GPU arch, TensorRT version).** Rebuild per target; store ONNX as source of truth; validate measured latency on each device rather than trusting that an engine "loaded."
9. **TensorRT is the right call on NVIDIA when latency or watts bind; ONNX Runtime when the fleet is portable; TensorRT-LLM when the workload is an LLM.** Name the cost — build complexity and non-portable engines — before pulling the lever.

## Further reading

- **NVIDIA TensorRT Developer Guide** — the authoritative reference for the builder, optimization profiles, the int8 calibrators, explicit (Q/DQ) versus implicit quantization, and version/hardware compatibility modes. Read the calibration and dynamic-shapes chapters in full before a production int8 build.
- **Szymon Migacz, "8-bit Inference with TensorRT" (NVIDIA GTC, 2017)** — the canonical explanation of entropy calibration and the KL-divergence threshold search; this is the *why* behind TensorRT's default int8 calibrator.
- **`trtexec` documentation** — the full flag reference for building and profiling from the command line; the fastest path to a measured number on a new chip.
- **NVIDIA Jetson / JetPack documentation and Jetson benchmarks** — power modes (`nvpmodel`), clock pinning (`jetson_clocks`), `tegrastats`, and the per-JetPack published benchmark numbers that illustrate the rebuild-per-version reality.
- **NVIDIA DLA documentation** — the Deep Learning Accelerator op-support matrix and restrictions; required reading before committing a model to the DLA.
- **TensorRT-LLM documentation** — the LLM-serving path with paged KV cache, in-flight batching, and FP8; the right tool when the edge workload is a language model.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame, [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) for the calibration numerics TensorRT applies, [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for the chips, [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for memory-bound versus compute-bound, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that sequences every lever. For the LLM path, [how the TensorRT inference compiler works end to end](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler).
