---
title: "The metrics that actually matter on-device: latency, memory, energy, and why FLOPs lie"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Learn to measure the numbers a phone, Jetson, or microcontroller actually feels — batch-1 tail latency, peak activation memory, joules per inference — and why a model with half the FLOPs can run slower."
tags:
  [
    "edge-ai",
    "model-optimization",
    "latency",
    "profiling",
    "inference",
    "efficient-ml",
    "roofline",
    "energy",
    "benchmarking",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-metrics-that-actually-matter-on-device-1.png"
---

A team I worked with spent three weeks redesigning their on-device image model. They swapped fat 3×3 convolutions for depthwise-separable ones, pruned a few channels, and proudly cut the theoretical compute roughly in half — from about 600 million to 300 million multiply-accumulate operations. The slide deck looked great. Then they ran it on the actual phone, a mid-range Android device they had to ship to, and the median inference time went *up*: from about 22 ms to about 29 ms. Same accuracy. Half the FLOPs. Thirty percent slower. The room went quiet.

This is the most common, most expensive mistake in edge ML, and almost everyone makes it once: **optimizing the number that is easy to compute instead of the number the device actually feels.** FLOPs are easy — you can count them statically, off a model summary, without ever touching hardware. Latency, peak memory, and energy are annoying — they depend on the chip, the runtime, the kernel library, the thermal state, the clock governor, and whether the layer is starved on memory bandwidth. So people optimize FLOPs and hope. The phone does not care about your FLOP count. It cares about how many bytes it had to drag across the memory hierarchy and whether its math units sat idle waiting for them.

![Before and after comparison showing a model with half the FLOPs running thirty percent slower on a phone because its depthwise layers are memory-bound](/imgs/blogs/the-metrics-that-actually-matter-on-device-1.png)

This post is the measurement foundation for the rest of the "Optimizing AI Models for the Edge" series. Before you quantize, prune, distill, or hand a model to a compiler, you have to be able to *measure the right thing* — otherwise every optimization is a coin flip and you will occasionally make things worse with full confidence. By the end you will be able to: define every metric that matters on-device with its real units; count FLOPs and MACs for a linear layer, a convolution, and attention from first principles; explain — with the arithmetic-intensity argument and the roofline — exactly *why* a low-FLOP model can be slow; write a latency-timing harness that is not lying to you; measure peak memory and energy honestly; and choose which single metric should drive *your* decision given *your* constraint. This is the "measure first" prerequisite that sits under the [taxonomy of compression techniques](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) and feeds straight into the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives).

## The one-sentence thesis: you cannot optimize what you mismeasure

Every chapter of this series is about moving a point on the accuracy–efficiency Pareto frontier — trading a little accuracy for a lot of speed, size, or battery. But "efficiency" is not one number. It is at least four families of number, each measured differently, each lying to you in a different way, and each mattering to a different stakeholder. The server team cares about throughput and cost-per-million-inferences. The mobile team cares about the 99th-percentile latency of a single inference and how much battery a feature drains. The microcontroller team cares whether the peak activation buffer fits in 256 KB of SRAM at all. If you collapse all of that into "FLOPs," you will routinely ship the wrong model.

The taxonomy of what to measure splits cleanly into four families, and it is worth fixing them in your head before we go deep on each.

![Tree diagram of the on-device metric taxonomy branching into latency, memory, compute, and energy families each with its unit](/imgs/blogs/the-metrics-that-actually-matter-on-device-2.png)

The four families are **latency** (time, in milliseconds or tokens per second), **memory** (space, in megabytes, with peak being the number that kills you), **compute** (work, in FLOPs or MACs — the seductive, misleading one), and **energy** (joules, milliwatt-hours, and the milliwatts that set your thermal ceiling). Compute is colored as a caution in that taxonomy on purpose: it is the only one of the four you can compute without hardware, which is exactly why it is the one people over-trust. The rest of this post defines each family precisely, shows you how to measure it without fooling yourself, and then spends real effort on the single most important idea: why compute (FLOPs) and time (latency) are *not the same axis*, and how to predict which one is going to bite you.

A note on scope before we dive in. Everything here is framed for inference, not training — the edge runs forward passes, not backward ones. And the running example for most of the post is a small image classifier in the MobileNet/ResNet family deployed to a phone CPU and NPU, with detours into LLM-specific metrics (prefill vs decode, KV-cache, tokens per second) because the whole series eventually puts a small language model on-device.

## Latency and throughput: the two are not interchangeable

The first metric anyone reaches for is latency. But "latency" alone is underspecified, and conflating it with throughput is the single most common benchmarking error after trusting FLOPs.

**Latency** is the wall-clock time for *one* inference, end to end, at **batch size 1**. It is what a user feels when they tap the shutter button and wait for the scene to be classified, or when they speak and wait for the wake word to fire. It is measured in milliseconds per inference (`ms/inf`). For an LLM it is often quoted as time-to-first-token (TTFT) and then time-per-output-token, or its reciprocal, tokens per second.

**Throughput** is how many inferences complete per unit time when you are allowed to batch and pipeline — measured in inferences per second (or tokens per second aggregated across a batch). It is what a *server* cares about because the server amortizes fixed costs across a big batch and is graded on cost-per-million-requests.

Here is the trap: **these two metrics pull in opposite directions, and the edge almost always lives in the batch-1 latency world while cloud benchmarks almost always quote batched throughput.** A GPU that does 10,000 images/second at batch 256 might take 8 ms to do a *single* image, because at batch 1 it cannot fill its thousands of ALUs and the per-launch overhead dominates. If your on-device feature processes one camera frame at a time, the batched-throughput number on the spec sheet is irrelevant — actively misleading, even. Always ask: *is this latency at batch 1, or throughput at batch N?* They differ by an order of magnitude or more, and only one of them is what your user experiences.

The relationship between them is worth making precise, because it explains *why* the gap exists rather than just asserting it. If a single inference at batch 1 takes latency $L_1$, and the hardware were perfectly serial, batch $N$ would take $N \cdot L_1$ and throughput would be flat at $1/L_1$. But hardware is not serial — it has parallel units that batch-1 work leaves idle. So in practice the time for a batch of $N$ is closer to $L_1 + (N-1)\cdot \delta$ where $\delta \ll L_1$ is the marginal per-item cost once the fixed launch and pipeline-fill overhead is amortized. Throughput is then $N / (L_1 + (N-1)\delta)$, which *rises* with $N$ toward the asymptote $1/\delta$. The whole point of batching is to make $N$ large enough that the fixed $L_1$ overhead is negligible per item — which is exactly the move a server makes and exactly the move an on-device single-frame feature *cannot* make. The edge is stuck paying the full fixed overhead on every single inference. That is not a tuning failure; it is the structural reason batch-1 latency is so much worse than the throughput-implied per-item time, and why you must never read a throughput spec as if it were a latency promise.

There is a corollary that bites teams who try to "fix" on-device latency by batching: on a thermally constrained device, a bigger batch raises peak memory (more activations live at once) and raises instantaneous power (more units active), which can trigger throttling *sooner* — so the batch that helped throughput can hurt the very tail latency you cared about. Batch size is a server lever; on-device it is usually pinned at 1 by the product (one camera frame, one user utterance) and trying to grow it fights the thermal budget.

### Why the tail matters more than the median on-device

You measure latency many times and you get a distribution, not a point. The honest way to report it is with percentiles:

- **p50 (median):** half of inferences are faster than this. The "typical" number.
- **p90:** nine out of ten are faster. The start of the tail.
- **p99:** ninety-nine out of a hundred are faster. The number that defines whether your feature feels reliable.

On a server, p99 matters because of queueing and fan-out: a request that touches ten services inherits the p99 of the slowest. On-device, p99 matters for a different and sneakier reason: **thermal throttling and the OS scheduler.** A phone runs your model on a SoC that shares power and thermal budget with the screen, the radio, and a dozen background apps. The first few inferences run at full clock; then the chip heats up, the governor drops the frequency to stay under its thermal ceiling, and your latency *climbs* — not on average, but at the tail, exactly when the user is doing something sustained like recording video. The scheduler can also preempt your thread to service a higher-priority task, adding a multi-millisecond spike. So a model with a 12 ms median can blow past a 16 ms real-time frame budget at p99 and drop frames, while its median looks perfectly healthy on your bench.

![Matrix showing cold first call, warm median, p90, and p99 latency against a sixteen millisecond budget with thermal throttling at the tail](/imgs/blogs/the-metrics-that-actually-matter-on-device-4.png)

The figure above makes the shape concrete with representative numbers for a vision model targeting 60 fps (a 16.7 ms frame budget). The cold first call at 140 ms is pure noise — JIT compilation, cold instruction cache, lazy kernel selection — and you must *never* count it. The warm p50 of 12 ms looks great. But the warm p99 of 24 ms, caused by thermal throttling under sustained load, is the number that decides whether the feature ships. If you optimize the median and ignore the tail, you will ship something that demos perfectly and then drops frames in the field after ninety seconds of recording. For real-time on-device work, **p99 is the metric, not p50.**

#### Worked example: prefill vs decode latency for an on-device LLM

LLMs split latency into two regimes that have completely different bottlenecks, and quoting one number hides this. Say you put a 3B-parameter model on a phone NPU at 4-bit weights (about 1.6 GB of weights). A user sends a 512-token prompt and you generate 128 tokens.

- **Prefill** processes all 512 prompt tokens *in parallel* in one big matrix-multiply-heavy pass. This is compute-bound — you are doing a lot of math per byte loaded — and it sets the time-to-first-token. Suppose prefill runs at 1,200 tokens/second, so TTFT ≈ 512 / 1200 ≈ 0.43 s.
- **Decode** generates the 128 output tokens *one at a time*, and each step must re-read the entire ~1.6 GB of weights from memory to produce a single token. This is brutally memory-bound. The decode rate is roughly (memory bandwidth) / (bytes read per token). On a phone with ~50 GB/s of usable bandwidth, that is ~50e9 / 1.6e9 ≈ 31 tokens/second, so the 128 output tokens take ≈ 4.1 s.

Total latency ≈ 0.43 + 4.1 ≈ 4.5 s. If you reported "tokens per second" as a single blended number you would compute (512+128) / 4.5 ≈ 142 tok/s and badly misrepresent the experience: the user waits 0.43 s to see *anything* and then reads output streaming at 31 tok/s. The optimization levers differ too — prefill wants more compute (or speculative tricks), decode wants less memory traffic per token (quantization, smaller KV-cache, batching). One metric, two regimes; report both.

## Model size on disk, parameter count, and the bytes actually loaded

The next family of confusion lives between three things that get used interchangeably and should not be: **parameter count**, **model size on disk**, and **bytes actually loaded at inference**.

**Parameter count** is how many learnable weights the model has — 25.6 million for ResNet-50, 4.2 million for MobileNetV2, 7 billion for a "7B" LLM. It is a property of the *architecture*, independent of how you store it. It is a useful proxy for capacity but tells you nothing directly about disk or memory until you fix a *data type*.

**Model size on disk** is bytes of the serialized file — the `.tflite`, `.onnx`, `.gguf`, or `.pt`. This is parameter count times bytes-per-parameter, plus a little overhead for the graph structure and metadata. The bytes-per-parameter is set by the data type:

$$\text{size}_\text{params} \approx N_\text{params} \times \frac{b}{8} \ \text{bytes}$$

where $b$ is bits per weight. So 7 billion parameters is about 28 GB in fp32 ($b=32$), 14 GB in fp16 ($b=16$), 7 GB in int8 ($b=8$), and roughly 3.5–4 GB in 4-bit ($b=4$, with a little extra for per-group scales). This single relationship is *why quantization is the headline lever for fitting LLMs on-device*: it directly multiplies the dominant term in the file size. Disk size is the metric your **app download size** and your **flash budget** care about. A microcontroller might have 1 MB of flash total; your model file competes with the rest of the firmware for it.

**Bytes actually loaded** is the subtler one, and it is the bridge to the energy and roofline discussion later. During inference, the runtime reads weights from storage into RAM (once, at load) and then reads them from RAM into the compute cores (every inference, or every token for an LLM). For a feed-forward CNN run once, you load each weight roughly once per inference. For an LLM in the decode phase, you re-read *every weight from memory for every single token* — which is exactly why decode is memory-bound and why a 4-bit model decodes roughly twice as fast as the same model in 8-bit even though they do the "same" math. The bytes-loaded-per-inference is the number that drives both decode latency and energy, and it is *not* visible in the parameter count.

A clean way to keep these straight: parameter count is capacity, disk size is download/flash, bytes-loaded-per-inference is speed-and-battery. Three different numbers, three different stakeholders, and an int8 model has the same parameter count as its fp32 origin while being four times smaller on disk and four times cheaper to load.

## Memory: parameters are the floor, peak activation is the cliff

Here is a failure mode that surprises people moving from cloud to edge: a model whose *parameters* fit comfortably in memory still crashes with an out-of-memory kill. The reason is that parameters are only one of the three memory costs, and they are usually not the one that OOMs you.

The three costs are:

1. **Parameter memory** — the weights, resident for the whole inference. $N_\text{params} \times b/8$ bytes, same as disk (minus compression). Fixed, predictable, you computed it above.
2. **Activation memory** — the intermediate tensors produced and consumed as data flows through the layers. A feature map after a conv, the hidden state between transformer layers. These are *transient*: created, used by the next layer, then (ideally) freed. But while they are live they occupy real RAM.
3. **Peak working memory** — the *maximum* amount of memory live at any single instant during the forward pass. This is parameters (always live) plus the largest set of activations that must coexist at one moment. **This peak is what the allocator must satisfy, and this peak is what triggers the OOM kill.**

![Stack diagram showing parameter memory as a fixed floor with input output tensors, live activations, and KV-cache stacked on top to form the peak working set](/imgs/blogs/the-metrics-that-actually-matter-on-device-3.png)

The figure shows why peak, not parameters, is the cliff. The 12 MB of int8 parameters is a flat floor that is always present. On top of it sits the input/output, then the *largest live activation set* — and for many vision models that single activation peak (an early high-resolution feature map, say 56×56×128 in fp32 = ~1.6 MB for one tensor, several of which may be live) dwarfs a small model's parameters. For an LLM, the **KV-cache** is the activation cost that grows without bound: every generated token appends a key and value vector for every layer, so the cache grows linearly with sequence length. A peak of ~38 MB OOMs a device that has the parameters fitting fine in 12 MB.

This is *the* metric for microcontrollers. A Cortex-M7 might have 512 KB of SRAM. Your int8 model parameters live in flash (read-only, cheap, abundant) — say 200 KB — but the activations must live in the precious SRAM tensor arena. If your peak activation set is 600 KB, the model literally cannot run, full stop, no matter how small the parameter count is. TFLite-Micro forces you to declare a fixed `tensor_arena` size at compile time precisely because peak activation memory is a hard, must-fit budget on these devices. The whole subfield of "memory planning" in edge compilers exists to *reduce the peak* by reusing buffers — freeing an activation the moment its last consumer is done so the next allocation can reuse the same bytes.

#### Worked example: peak memory for a MobileNetV2 on a phone vs an MCU

Take MobileNetV2 at 224×224 input, the canonical mobile classifier.

- **Parameters:** ~3.5M. In fp32 that is ~14 MB; in int8, ~3.5 MB.
- **Peak activation (fp32):** dominated by the early high-resolution stages. The first inverted-residual block expands to 96 channels at 112×112 = ~1.2M elements = ~4.8 MB for a single fp32 tensor, and the block keeps the input and the expanded tensor live simultaneously, pushing the peak into the ~5–10 MB range depending on the implementation.
- **On a phone (say 6 GB RAM):** 14 MB params + ~10 MB peak activation is a rounding error. Memory is not your constraint here; latency is.
- **On a Cortex-M7 (512 KB SRAM, 2 MB flash):** the fp32 model is dead on arrival — 14 MB params won't even fit in flash. Quantize to int8: params drop to ~3.5 MB (still too big for 2 MB flash — you'd need a smaller backbone or more aggressive compression), and the peak activation, even quantized to ~2.5 MB, vastly exceeds 512 KB of SRAM. The lesson: on an MCU the binding constraint is peak activation memory in SRAM, and it forces architecture choices (smaller input resolution, fewer channels, models like MCUNet designed for the SRAM budget) that have nothing to do with parameter count. You can have a 200 KB model that won't run because its 700 KB activation peak doesn't fit.

### KV-cache: the activation cost that never stops growing

For LLMs the dominant *transient* memory cost is the KV-cache, and it deserves its own formula because it is the number that decides how long a context your phone can actually hold. During autoregressive decode, each layer caches the key and value vectors for every token seen so far, so it never has to recompute them. The cache size is:

$$\text{KV bytes} = 2 \times n_\text{layers} \times n_\text{tokens} \times d_\text{model} \times \frac{b}{8}$$

The leading $2$ is one tensor for keys and one for values. Plug in a 7B-class model — say 32 layers, $d_\text{model} = 4096$ — at fp16 ($b=16$) for a 2,048-token context:

$$\text{KV bytes} = 2 \times 32 \times 2048 \times 4096 \times 2 \approx 2.15\ \text{GB}$$

Read that twice: the KV-cache for a single 2K-token conversation is over 2 GB *on top of* the ~14 GB of fp16 weights — and it grows *linearly with every token generated*. This is why on-device LLMs use grouped-query attention (which shares K/V across query heads and shrinks the cache by the grouping factor), quantize the KV-cache itself to int8 or int4, and cap context length aggressively. Parameter count tells you none of this; the KV-cache is pure activation memory, it scales with the conversation rather than the architecture, and on a memory-tight phone it is frequently the thing that decides whether a long chat OOMs. The peak-memory metric, not the model size, is what you budget against for on-device generation.

## Compute: FLOPs, MACs, and how to count them

Now the seductive metric. **FLOPs** = floating-point operations. **MACs** = multiply-accumulate operations. One MAC is a multiply followed by an add — $a \cdot b + c$ — and since each is a floating-point operation, **1 MAC ≈ 2 FLOPs**. This factor of two is the single most common source of "your FLOP count is double mine" arguments; tools like `thop` and `fvcore` report MACs (which they sometimes confusingly label "FLOPs"), `ptflops` reports MACs, and papers are split. Always state which you mean. Throughout this post I'll count MACs (the multiply-accumulates) and convert to FLOPs as $2\times$ when needed.

Let's count them from first principles for the three layer types that make up almost everything.

### A dense / linear layer

A linear layer computes $y = Wx$ where $W$ is $M \times K$ (output dim $M$, input dim $K$) and $x$ has $K$ elements producing $M$ outputs. Each of the $M$ outputs is a dot product over $K$ inputs — $K$ multiply-accumulates each. So:

$$\text{MACs} = M \cdot K, \qquad \text{FLOPs} = 2 \cdot M \cdot K$$

For a batch (or sequence) of $B$ tokens it is $2 \cdot B \cdot M \cdot K$. This $2MNK$ form (with $N$ the batch/sequence dimension) is the workhorse formula — every fully-connected layer, every projection in a transformer, is an instance of it. A 4096→4096 projection on one token is $2 \cdot 4096 \cdot 4096 \approx 33.6$ MFLOPs. The weight bytes loaded are $M \cdot K \cdot (b/8)$ — hold that thought, it's the numerator of arithmetic intensity.

### A convolution

A 2D convolution with $C_\text{in}$ input channels, $C_\text{out}$ output channels, a $K_h \times K_w$ kernel, producing an output of spatial size $H_\text{out} \times W_\text{out}$, costs one dot product of length $C_\text{in} \cdot K_h \cdot K_w$ per output element, and there are $H_\text{out} \cdot W_\text{out} \cdot C_\text{out}$ output elements:

$$\text{MACs}_\text{conv} = \underbrace{H_\text{out} \cdot W_\text{out} \cdot C_\text{out}}_{\text{output elements}} \times \underbrace{C_\text{in} \cdot K_h \cdot K_w}_{\text{work per element}}$$

Concretely, a 3×3 conv with 128 input and 128 output channels over a 56×56 output map is $56 \cdot 56 \cdot 128 \cdot 128 \cdot 3 \cdot 3 \approx 462$ MMACs ≈ 925 MFLOPs. That single layer is heavier than the whole MobileNetV2 it might appear in — which is exactly the pressure that pushed mobile architectures toward cheaper convolutions.

### A depthwise-separable convolution — and the first crack in FLOPs

A **depthwise-separable** conv replaces the standard conv with two cheaper steps: a *depthwise* conv that applies one $K_h \times K_w$ filter per input channel (no mixing across channels), then a $1 \times 1$ *pointwise* conv that mixes channels. The MAC counts:

$$\text{MACs}_\text{depthwise} = H_\text{out} \cdot W_\text{out} \cdot C_\text{in} \cdot K_h \cdot K_w$$
$$\text{MACs}_\text{pointwise} = H_\text{out} \cdot W_\text{out} \cdot C_\text{in} \cdot C_\text{out}$$

For the 56×56, 128→128, 3×3 example: depthwise is $56 \cdot 56 \cdot 128 \cdot 9 \approx 3.6$ MMACs and pointwise is $56 \cdot 56 \cdot 128 \cdot 128 \approx 51.4$ MMACs, total ~55 MMACs versus the standard conv's 462 MMACs. **An 8.4× FLOP reduction.** This is exactly the move that team made — and exactly where they got burned, because the depthwise part has tiny FLOPs but also tiny *arithmetic intensity*, which we are about to define and which is the real reason it ran slow.

### Attention

For a self-attention layer over a sequence of length $n$ with model dimension $d$, the two big costs are the QK$^\top$ score matrix and the score-times-V output. Scores are $n \times n$, each an inner product of length $d$:

$$\text{MACs}_{QK^\top} = n^2 \cdot d, \qquad \text{MACs}_{\text{scores}\cdot V} = n^2 \cdot d$$

so the attention core is $\approx 2 n^2 d$ MACs — the famous **$O(n^2 d)$** term that makes long context expensive. The projections (Q, K, V, and output) add $4 \cdot n \cdot d^2$ MACs, which dominates at short sequences; the $n^2$ term takes over once $n \gtrsim d$. This is why context length is the lever everyone fights over on-device: doubling $n$ quadruples the attention compute *and* the KV-cache memory.

#### Worked example: when does attention's quadratic term actually bite?

The crossover where the $n^2 d$ score term overtakes the $4 n d^2$ projection term is when $2 n^2 d > 4 n d^2$, i.e. $n > 2d$. For a model with $d = 4096$, that's $n > 8192$ tokens. So below ~8K context the attention layer's compute is dominated by the *projections* (which are plain matmuls and scale linearly with $n$), and the quadratic term is a minor contributor. People often quote "attention is $O(n^2)$, that's the problem" and then optimize the score computation — but at the 512-to-2048-token contexts most on-device assistants run, the projections and the per-token weight reload (the memory-bound decode cost from earlier) are the real bottleneck, not the $n^2$ math. The metric framework saves you here: count the FLOPs *by term*, check which dominates at your actual sequence length, and you'll often find the quadratic boogeyman is asleep and the memory-bound decode is the one eating your tokens-per-second. Optimize the bottleneck that's awake.

## Why FLOPs lie: arithmetic intensity and the roofline

Now the heart of the post. We have a clean, correct FLOP count. Why does it fail to predict latency? Because **a processor cannot do math on data it has not yet loaded from memory**, and for many layers the loading is the bottleneck, not the math. The bridge concept is **arithmetic intensity**.

> **Arithmetic intensity** $I$ is the ratio of compute to memory traffic for an operation: how many FLOPs you do per byte you move from memory.
> $$I = \frac{\text{FLOPs}}{\text{bytes moved}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]$$

A processor has two relevant speed limits: a **peak compute rate** $\pi$ (FLOP/s, set by how many multiply-accumulate units it has and how fast they clock) and a **peak memory bandwidth** $\beta$ (bytes/s, how fast it can stream data from DRAM). For an operation of intensity $I$, the achievable rate is bounded by *both*, and the actual ceiling is the smaller of the two:

$$\text{attainable FLOP/s} = \min\big(\pi,\ \beta \cdot I\big)$$

This is the **roofline model** (Williams, Waterman, Patterson, 2009). It splits operations into two regimes by comparing $I$ to the hardware's **ridge point** $I^* = \pi / \beta$:

- If $I > I^*$: you are **compute-bound**. The math units are the limit; you are doing enough work per byte to keep them fed. Here, and *only* here, **FLOPs predict latency** — halving FLOPs roughly halves time.
- If $I < I^*$: you are **memory-bound**. You finish the math faster than memory can deliver the next operands, so the ALUs sit idle waiting on DRAM. Time is set by `bytes / β`, *not* by FLOPs. Halving the FLOPs of a memory-bound layer barely moves its latency, because the memory traffic didn't change.

![Before and after diagram contrasting a compute-bound convolution where FLOPs predict time against a memory-bound depthwise convolution where cutting FLOPs gives no speedup](/imgs/blogs/the-metrics-that-actually-matter-on-device-5.png)

Let's make this rigorous for our two convs. Consider the 56×56, 128→128, 3×3 layer in fp32 (4 bytes/element).

**Standard conv.** FLOPs ≈ 925M (from above). Bytes moved (a back-of-envelope: read input, read weights, write output): input ~$56\cdot56\cdot128\cdot4$ ≈ 1.6 MB, weights $128\cdot128\cdot9\cdot4$ ≈ 0.6 MB, output ~1.6 MB, total ~3.8 MB. Intensity $I \approx 925\text{M} / 3.8\text{M} \approx 243$ FLOP/byte. (With cache reuse of weights and input across output positions, effective intensity is even higher.) This is a *high* intensity — comfortably above the ridge point of most edge chips — so it is compute-bound and its FLOPs predict its time.

**Depthwise conv (the depthwise step alone).** FLOPs ≈ $2 \cdot 3.6$M ≈ 7.2M. Bytes moved: input ~1.6 MB, weights $128\cdot9\cdot4$ ≈ 4.6 KB (tiny — that's the point of depthwise, one filter per channel), output ~1.6 MB, total ~3.2 MB. Intensity $I \approx 7.2\text{M} / 3.2\text{M} \approx 2.3$ FLOP/byte. This is a *catastrophically low* intensity. The depthwise conv reads a full-size input and writes a full-size output but does almost no math in between — it is memory-bound by a mile.

Now plug into the ceiling. Suppose the chip has $\pi = 200$ GFLOP/s and $\beta = 20$ GB/s, so the ridge point $I^* = 200/20 = 10$ FLOP/byte.

- Standard conv ($I=243 \gg 10$): compute-bound, runs near 200 GFLOP/s, time ≈ $925\text{M} / 200\text{G} ≈ 4.6$ ms.
- Depthwise step ($I=2.3 < 10$): memory-bound, runs at $\beta \cdot I = 20\text{G} \cdot 2.3 = 46$ GFLOP/s — *less than a quarter of peak* — and its time is set by bytes: $3.2\text{M} / 20\text{G} ≈ 0.16$ ms.

The depthwise step is still faster in absolute terms here because it moves and computes far less overall. The *problem* shows up when depthwise-separable layers stack up and each one underutilizes the hardware: you spend a large fraction of total inference time on layers that are running the ALUs at 20% efficiency, idling on memory. The whole network's effective throughput collapses even though its total FLOP count went down. That is precisely how a half-the-FLOPs model ends up slower: you traded high-intensity, ALU-saturating convolutions for a pile of low-intensity, bandwidth-starved ones, and the chip spent most of its time waiting on DRAM. The FLOP count celebrated; the roofline wept.

There are several *other* reasons FLOPs mispredict latency, all of which compound the intensity story:

- **Per-op launch overhead.** Every operator has fixed dispatch cost — kernel launch, descriptor setup, synchronization. A model split into many tiny ops (lots of depthwise + 1×1 pairs, lots of reshapes) pays this overhead per op regardless of FLOPs. Fusing ops (a compiler job) is often a bigger win than cutting FLOPs.
- **Unsupported ops fall back to CPU.** If the NPU delegate doesn't support an op (an exotic activation, a custom layer, a weird padding), the runtime silently falls back to CPU and pays a *round-trip*: copy the tensor off the accelerator, run on CPU, copy back. One unsupported op in the middle of a graph can cost more than all the FLOPs around it. Your FLOP count has no idea this happened.
- **Parallelism limits at batch 1.** Wide hardware (many cores, big SIMD, tensor cores) needs enough independent work to fill it. At batch 1 with small spatial dimensions, a layer may not have enough parallel work to occupy the chip, so it runs at a fraction of peak no matter how many FLOPs it has.
- **Data type and kernel quality.** An int8 conv with a well-tuned kernel can be 3× faster than the "same FLOPs" in fp32; a poorly-tuned int8 kernel can be *slower* than fp16. FLOPs are blind to which kernels exist and how good they are.

This is the central message of the ShuffleNet v2 paper (Ma et al., 2018), whose "practical guidelines for efficient CNN design" argued — with direct measurements on a phone and GPU — that FLOPs (which they call indirect metric) routinely mispredict latency, and that you should design and select against *measured latency on the target* plus memory-access cost, not FLOP count. It is the same lesson, ten years before it was fashionable to relearn it.

We'll go much deeper on the roofline — drawing the actual ceilings, placing your layers on the chart, and using it to *decide* which lever to pull — in [the roofline model post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives). For now the operational takeaway is: **before you trust a FLOP reduction, ask whether the layers you're cutting were compute-bound. If they were memory-bound, the FLOP cut buys you nothing and may cost you, because you might be replacing a few high-intensity ops with many low-intensity ones.**

## Energy and power: the metric that tracks bytes, not FLOPs

For anything battery-powered — a phone, a wearable, a drone, a sensor node — energy is often the *real* constraint, and it has its own counterintuitive physics. Two quantities:

- **Power** is the instantaneous rate of energy use, in watts (W) or milliwatts (mW). It sets the thermal ceiling and the throttling behavior we saw in the tail-latency section.
- **Energy** is power integrated over time, in joules (J) or, for battery talk, milliwatt-hours (mWh). Energy per inference is what determines how many inferences you get per charge.

The crucial fact, and the reason this section exists right after the FLOPs discussion: **energy tracks memory movement far more than it tracks arithmetic.** This comes from the physics of the hardware, quantified in Mark Horowitz's widely-cited energy table (his 2014 ISSCC "Computing's Energy Problem" keynote). The numbers, in a 45 nm-ish process, are roughly:

| Operation | Approx. energy |
| --- | --- |
| 32-bit int ADD | ~0.1 pJ |
| 32-bit float MAC | ~1–4 pJ |
| 8 KB SRAM read | ~5–10 pJ |
| 32-bit DRAM access | ~640–1300 pJ |

Read that table again. **A DRAM access costs hundreds to thousands of times more energy than the arithmetic operation it feeds.** Moving a 32-bit operand from off-chip DRAM is on the order of 1000 pJ; the multiply-accumulate that consumes it is a few pJ. So the energy of an inference is dominated by *how many bytes you haul across the memory hierarchy*, especially how many times you go all the way to DRAM. And remember: the metric that captures bytes-per-FLOP is exactly arithmetic intensity. A low-intensity, memory-bound layer is not just slow — it is *expensive in joules*, because it spends its time and energy on data movement.

![Stack diagram of where a joule goes showing arithmetic costing fractions of a picojoule while DRAM access costs hundreds of picojoules and the radio costs millijoules](/imgs/blogs/the-metrics-that-actually-matter-on-device-8.png)

The figure stacks the energy hierarchy and adds the one that dwarfs everything else for connected devices: **the radio.** Transmitting a result over Wi-Fi or cellular costs on the order of millijoules — thousands of times a DRAM access, millions of times a MAC. This is the entire energy argument *for* on-device inference: a few hundred microjoules to run a model locally versus a few millijoules to ship the input to a server and stream the answer back. It also tells you that for an always-on connected sensor, the dominant lever is "don't transmit," not "fewer FLOPs."

Two practical consequences for the optimizer:

1. **Quantization saves energy primarily by moving fewer bytes,** not by cheaper arithmetic — int8 weights are half the bytes of fp16, so you make half the memory accesses, and memory accesses are where the joules are. (The cheaper int8 math is a bonus, not the main event.)
2. **Keeping data on-chip is the highest-leverage energy optimization.** Operator fusion that keeps an activation in SRAM instead of spilling it to DRAM between two ops can save more energy than a large FLOP reduction, because each avoided DRAM round-trip is worth ~100× the math.

#### Worked example: mWh per inference and battery life on a phone

Suppose your int8 vision model runs in 12 ms (p50) on a phone NPU and the NPU + memory subsystem draws ~700 mW while doing so (a plausible mid-range figure; the SoC as a whole might draw 2–3 W under load). Energy per inference:

$$E = P \cdot t = 0.7\,\text{W} \times 0.012\,\text{s} = 8.4\,\text{mJ} = 8.4\,\text{mJ} \times \frac{1\,\text{mWh}}{3.6\,\text{J}} \approx 0.0023\,\text{mWh}$$

So ~2.3 microwatt-hours per inference. A phone battery is ~15 Wh = 15,000 mWh. Ignoring everything else, that is ~6.4 million inferences per charge from this model alone — clearly not the constraint for an occasional tap. But flip it to an *always-on* feature running 30 inferences/second (a real-time camera filter): that's $30 \times 0.0023 \approx 0.069$ mWh/s = 248 mWh/hour, which would drain ~1.6% of the battery per hour from this model alone — now it matters, and it competes with the screen and radio. The same model is "free" in one usage pattern and a battery hog in another. **Energy is a rate-times-duty-cycle question, not a per-inference constant.** And note: had you optimized this model for FLOPs into a memory-bound shape, the *per-inference energy could rise* even as FLOPs fell, because you'd move more bytes per inference.

## How to measure each metric honestly — the practical flow

Knowing the metrics is half of it. Measuring them without lying to yourself is the other half, and it is where most benchmark numbers go wrong. Here is the honest method for each.

### Counting FLOPs and parameters

This one is static and easy — use a tool, don't count by hand for a real model. `thop`, `fvcore`, and `ptflops` all walk the graph and tally MACs and params. Watch the MAC-vs-FLOP factor of two.

```python
import torch
import torchvision.models as models
from thop import profile, clever_format

model = models.mobilenet_v2(weights=None).eval()
dummy = torch.randn(1, 3, 224, 224)  # batch=1, the on-device reality

macs, params = profile(model, inputs=(dummy,), verbose=False)
flops = 2 * macs  # thop reports MACs; 1 MAC = 2 FLOPs
macs_s, params_s, flops_s = clever_format([macs, params, flops], "%.2f")
print(f"Params: {params_s}   MACs: {macs_s}   FLOPs: {flops_s}")
# Params: 3.50M   MACs: 314.32M   FLOPs: 628.63M  (approx)
```

`fvcore` is the more careful choice for transformers because it has analytic handlers for attention and gives a per-module breakdown so you can see *which* layers carry the compute:

```python
from fvcore.nn import FlopCountAnalysis, flop_count_table

flops = FlopCountAnalysis(model, dummy)
print(flop_count_table(flops, max_depth=2))  # per-module MACs, find the hot blocks
```

The per-module table is the useful part: it tells you *where* the FLOPs are, which you then cross-check against *where the latency is* — and the gap between those two rankings is the whole point of this post.

### A correct latency-timing loop

This is the code that everyone gets wrong. The mistakes are: timing the cold first call, forgetting to synchronize the device (so you time the *launch* of async kernels, not their completion), including data-loading or pre/post-processing in the timer, reporting the mean (which one throttle spike poisons), and letting the clock governor change frequency mid-measurement. The correct pattern is **warm up, lock clocks, synchronize, time many runs, report the median and tail.**

![Timeline of a correct latency timing loop with warm-up, clock locking, device sync, N timed runs, and reporting the median plus tail percentiles](/imgs/blogs/the-metrics-that-actually-matter-on-device-6.png)

```python
import torch, time, numpy as np

def benchmark_latency(model, example_input, n_warmup=20, n_runs=200, device="cuda"):
    model = model.to(device).eval()
    x = example_input.to(device)

    # 1) WARM-UP: triggers JIT/autotuning, warms caches, lets the
    #    clock governor settle. Throw these away.
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()  # ensure warm-up actually finished

    # 2) TIMED RUNS: sync the device around EACH run so we time the
    #    kernel, not the async launch. Record per-run times.
    times = []
    with torch.inference_mode():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()  # CRITICAL: wait for completion
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

    t = np.array(times)
    return {
        "p50_ms": float(np.percentile(t, 50)),
        "p90_ms": float(np.percentile(t, 90)),
        "p99_ms": float(np.percentile(t, 99)),
        "mean_ms": float(t.mean()),  # report but don't trust for tail
    }
```

The non-obvious lines:

- **`torch.cuda.synchronize()` is not optional.** CUDA kernels launch asynchronously — `model(x)` returns *before* the GPU finishes. Without the sync after the call, `perf_counter` measures launch overhead (microseconds) instead of execution (milliseconds), giving you absurd "0.1 ms" numbers. On CPU you don't need it, but you do need to make sure nothing is lazily deferred. On accelerators with their own profilers (NPUs via TFLite/ONNX Runtime), use the runtime's own timing, which handles the sync internally.
- **Report the median, not the mean.** A single throttle spike or scheduler preemption inflates the mean and makes runs incomparable. The median is robust; pair it with p90/p99 to see the tail explicitly.
- **Lock clocks for *reproducibility*.** On a Jetson, `sudo jetson_clocks` pins the clocks to max so you measure the model, not the governor's mood. On a phone, you can't usually lock clocks, so you instead measure under a *controlled thermal state* (let it cool, or measure sustained to capture the throttled tail on purpose) and you report both the steady warm number and the sustained-load tail.
- **Exclude data-loading and pre/post-processing from the timer** unless you are explicitly benchmarking the full pipeline. Time the model, then separately time the pipeline; conflating them hides where the cost is.

For the real target, use the runtime's own benchmark tool rather than a Python loop — they handle device sync, threading, and delegates correctly:

```bash
# TFLite benchmark tool: batch=1, single-thread, NNAPI delegate (NPU),
# warm-up runs, reports avg + percentiles. This is the on-device truth.
./benchmark_model \
  --graph=model_int8.tflite \
  --num_threads=1 \
  --use_nnapi=true \
  --warmup_runs=20 \
  --num_runs=200 \
  --enable_op_profiling=true   # per-op latency: find the real hot ops

# ONNX Runtime perf test: same idea, with the chosen execution provider.
onnxruntime_perf_test -e nnapi -r 200 -w 20 -m times model_int8.onnx
```

The `--enable_op_profiling` flag is the gold: it gives **per-operator latency on the actual device**, which is the measurement that exposes the FLOPs-vs-latency gap directly. You will routinely see an op that is 2% of the FLOPs taking 30% of the time (memory-bound, or falling back to CPU), and another that is 40% of the FLOPs taking 10% of the time (compute-bound and well-tuned). That ranking is what you optimize against — never the FLOP ranking alone.

#### Worked example: the four ways a benchmark lies, quantified

To make the timing mistakes concrete, here is the same MobileNetV2 measured four ways on the same laptop GPU, showing how each mistake distorts the number.

- **Mistake 1 — timing the cold first call.** You run `model(x)` once and time it: **140 ms**. That's JIT compilation, kernel autotuning, and cold caches — a one-time cost the user pays only on the very first inference after launch (and you can hide it behind a splash screen). Reporting 140 ms as "the latency" overstates it ~15×.
- **Mistake 2 — forgetting `synchronize`.** You loop without the post-call sync and time: **0.4 ms**. This is *too fast to be real* — it's the async launch returning before the GPU has done anything. Telltale sign: a latency that doesn't change when you make the model bigger. Understates the true number ~30×.
- **Mistake 3 — reporting the mean over a noisy run.** You time 200 runs but compute the mean: **18 ms**, where the median is 9 ms. One thermal spike or a background process preempting your thread for 200 ms drags the mean up while the median stays honest. Overstates the typical case ~2×.
- **Mistake 4 — including the data pipeline.** You time `preprocess(img); model(x); postprocess(out)` together: **34 ms**, of which the model is 9 ms and the JPEG decode plus resize plus normalize is 25 ms. Now you "optimize the model" and the number barely moves, because the model was never the bottleneck. The fix is to time the model in isolation *and* the pipeline end to end, then attribute.

The correct number — warm, synced, median of 200, model only — is **9 ms**. Notice that all four mistakes are *plausible*: each produces a number a tired engineer would write in a slide. The harness above eliminates all four, which is why the boilerplate (warm-up, sync, median, isolate) is non-negotiable rather than optional polish.

### Measuring peak memory

On a GPU, PyTorch tracks the high-water mark for you:

```python
import torch

torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    _ = model(x)            # one forward pass at batch=1
torch.cuda.synchronize()
peak_bytes = torch.cuda.max_memory_allocated()
print(f"Peak working memory: {peak_bytes / 1e6:.1f} MB")
```

`max_memory_allocated` returns the **peak** allocator high-water mark over the window — exactly the activation-plus-parameter peak that determines whether you OOM. Reset it before the pass so a previous run's peak doesn't contaminate the number. On CPU, use `tracemalloc` or `psutil` RSS deltas (coarser, includes framework overhead). On a phone, the runtime reports its arena size; on TFLite-Micro you *declare* the tensor arena and the build fails or asserts if peak activation exceeds it — the most honest peak-memory check there is, because it's a hard compile-time budget.

### Measuring energy and power

Energy is the hardest to measure honestly because you need to read the rail, not estimate from FLOPs. Options, roughly from cheapest to most accurate:

- **Battery stats (coarse, free).** Android's `batterystats` / Battery Historian, or iOS Energy Log, attribute energy to your app over a session. Good for "did this feature get cheaper" deltas; too noisy for per-inference numbers.
- **On-SoC power telemetry (good, free on dev boards).** `tegrastats` on Jetson reports per-rail power (GPU, CPU, SoC, DRAM) in mW at ~100 ms cadence — run it while you loop inference and integrate. On a Mac, `sudo powermetrics --samplers cpu_power,gpu_power -i 200` reports package power you can integrate over a timed run. On some Android SoCs, ODPM rails are exposed via the kernel.

```bash
# Jetson: sample power while the model loops in another shell, then integrate.
tegrastats --interval 100 | grep -o "POM_5V_IN [0-9]*" 
# (each line gives instantaneous mW; mean(mW) * runtime(s) / 3600 = mWh)

# macOS: package power during a timed inference loop.
sudo powermetrics --samplers cpu_power,gpu_power -i 200 -n 50
```

- **External power monitor (gold standard).** A Monsoon High Voltage Power Monitor or an INA219/INA260 shunt on the device's supply rail measures *true* energy at the battery terminals, including the radio and screen, at sub-millisecond resolution. This is what you use for a paper or a shipping power budget. The method: run the model in a tight loop for a fixed wall-clock window, measure average power, subtract the idle baseline (measure idle power separately), multiply by inference time, divide by inference count.

The honest energy-per-inference is `(P_active - P_idle) * t_inference`, and you *must* subtract the idle baseline or you'll attribute the screen and OS to your model. Report it as "incremental energy of running the model," with the device, the rail measured, and whether the radio was on.

## A worked example: all metrics side by side, and the trap

Let's pull it together on one model and one named target, then spring the FLOPs trap with real-shaped numbers.

#### Worked example: ResNet-18 vs a depthwise-heavy variant on a Raspberry Pi 5 CPU

Take ResNet-18 (standard 3×3 convs, compute-heavy) and a hand-built "efficient" variant where we replaced the main 3×3 convs with depthwise-separable blocks, tuned so it matches ResNet-18's top-1 accuracy on the target task (say both at 71.0% — we held accuracy fixed on purpose, because that's the honest comparison: equal accuracy, which one is better on the device?). Target: Raspberry Pi 5, 4× Cortex-A76 @ 2.4 GHz, measured single-threaded, fp32, with the correct warm-up + median harness above.

| Metric | ResNet-18 | Depthwise variant | Winner |
| --- | --- | --- | --- |
| Top-1 accuracy | 71.0% | 71.0% | tie (held fixed) |
| Parameters | 11.7M | 4.0M | depthwise (2.9× fewer) |
| FLOPs (MACs ×2) | ~3.6 GFLOPs | ~1.1 GFLOPs | depthwise (3.3× fewer) |
| Model size (fp32) | ~45 MB | ~16 MB | depthwise |
| Peak activation mem | ~12 MB | ~9 MB | depthwise (slightly) |
| **Latency p50 (RPi5 CPU)** | **~95 ms** | **~120 ms** | **ResNet-18 (faster!)** |
| Latency p99 | ~105 ms | ~140 ms | ResNet-18 |
| Energy / inference | ~0.45 J | ~0.55 J | ResNet-18 |

Look at that table and feel the trap close. Every *static* metric says the depthwise variant is better: fewer params, a third of the FLOPs, smaller file, lower peak memory. If you had stopped at the model summary — as that team did — you would ship the depthwise variant and tell your manager you made it 3× cheaper. But on the actual Cortex-A76 CPU it is **~25% slower and uses ~20% more energy at equal accuracy.**

Why? The arithmetic-intensity story, exactly as derived. ResNet-18's 3×3 convs are high-intensity, compute-bound layers that the CPU's NEON SIMD units and cache hierarchy chew through efficiently — high ALU utilization, weights reused heavily across spatial positions. The depthwise convs are low-intensity, memory-bound layers: they stream the full feature map in and out of cache while doing almost no math per byte, so the CPU spends most of its time waiting on memory, and the many small depthwise + 1×1 ops each pay per-op overhead. The FLOP count fell by 3.3×; the *memory traffic* and *op count* did not, and on this CPU memory traffic and op overhead are what set the time. The energy followed the bytes, not the FLOPs, so the "cheaper" model drained more battery.

The fix is not "always use ResNet" — depthwise architectures genuinely win on hardware with the right kernels (a well-tuned mobile NPU, a GPU with good depthwise support). The fix is: **measure latency and energy on the actual target before you believe a FLOP reduction.** The right model is target-dependent, and only measurement tells you which world you're in. (This is also a preview of why "efficient architectures" is one of the four levers but only pays off when matched to the runtime and hardware — a thread we pick up across the series.)

## Stress-testing the decision: walking through the real problem

Let's stop being abstract and reason through the actual engineering problem the way you'd face it on a Friday afternoon with a ship date. You have a model, a target chip, and a complaint ("the feature feels laggy" or "the app is too big" or "it kills the battery"). What's your move?

**Step 1 — translate the complaint into a metric and a budget.** "Laggy" for a real-time camera feature means a p99 latency above the frame budget — quantify it: 60 fps is 16.7 ms, so the budget is "p99 < 16.7 ms under sustained load." "Too big" means model size on disk above the store limit. "Kills the battery" means mWh/hour above what the product can spend. You cannot proceed until the complaint is a number with a threshold. This is the step everyone skips, and skipping it is how you end up optimizing FLOPs: FLOPs is the metric you reach for when you haven't pinned down which number actually matters.

**Step 2 — measure the current state on the target, honestly.** Run the warm-up + sync + median + p99 harness. Turn on per-op profiling. Now you have a ranked list of *where the time actually goes* — and crucially, you compare it to the FLOP ranking from `fvcore`. The gap between those two lists is your map. An op that is high in latency but low in FLOPs is memory-bound or falling back to CPU; an op that is high in both is genuinely compute-heavy. You optimize against the latency ranking, using the FLOP ranking only to diagnose *why*.

**Step 3 — diagnose the regime with arithmetic intensity.** For the hottest ops, estimate $I = \text{FLOPs}/\text{bytes}$. If the hot op is compute-bound ($I > I^*$), a FLOP-reducing change (fewer channels, lower-rank factorization) will actually help. If it's memory-bound ($I < I^*$), FLOP cuts are wasted; the levers are quantization (move fewer bytes), fusion (avoid DRAM round-trips), or a different op the runtime executes better. This single diagnostic redirects you from the wrong lever to the right one.

Now stress-test that reasoning against the cases that break naive intuition:

- **"What if the NPU doesn't support an op and it falls back to CPU?"** Then a tiny-FLOP op (say a custom activation or an unusual reshape) can dominate latency because of the off-accelerator copy round-trip. The per-op profiler shows this as a latency spike with no FLOP justification, often with the op running on a different backend in the trace. The fix is to replace the unsupported op with a supported equivalent (a plain ReLU6 instead of an exotic activation, a supported padding mode) so the whole graph stays on the NPU — a change that cuts latency dramatically while *raising* FLOPs slightly. FLOPs would have told you to do the opposite.
- **"What if the calibration set is tiny and int8 latency looks great but accuracy tanks?"** Latency is honest (the int8 kernels are fast) but your *guardrail* metric (accuracy) failed. This is why you never optimize a single metric without a guardrail: the right answer is "int8 latency is great, but accuracy dropped 4 points, so the calibration set or scheme is the problem" — not "int8 is bad." The metric framework keeps you from blaming the wrong thing.
- **"What if the model is memory-bound and I quantize?"** Best case: quantization is *exactly* the right lever, because it halves (int8) or quarters (int4) the bytes moved, directly attacking the bandwidth bottleneck that FLOP cuts couldn't touch. This is why quantization so often beats architectural FLOP reduction on memory-bound, bandwidth-starved layers and on LLM decode — it fixes the actual bottleneck.
- **"What if I drop to int4 to go faster?"** Now you stress two metrics at once: latency keeps improving (even fewer bytes), but accuracy degradation accelerates non-linearly, and below int4 the per-group scale overhead and dequant cost can start eating the gains. The metric to watch flips from latency to accuracy; the framework tells you which guardrail is now binding.

The discipline running through every case: **a metric is only meaningful against a budget, you always carry a guardrail (usually accuracy), and you diagnose the regime before picking the lever.** That is the entire difference between engineering and guessing, and it is why this measurement post comes before any technique post in the series.

## Results tables: what each metric tells you, and fp32 vs int8

Two tables you should internalize. The first is a cheat-sheet for *which metric answers which question and how it lies*; the second is the canonical before→after that quantization produces, which is the single most common optimization in the series.

![Matrix mapping each metric to what it tells you, how to measure it, and its classic gotcha across latency, memory, size, energy, and FLOPs](/imgs/blogs/the-metrics-that-actually-matter-on-device-7.png)

The figure above is the decision cheat-sheet rendered; here it is as a table with more room.

| Metric | What it tells you | How to measure (honestly) | The gotcha |
| --- | --- | --- | --- |
| Latency p50 / p99 | Does it feel fast for one user, including the tail? | Warm-up + sync + median + p99 loop; runtime benchmark tool; per-op profiling | The tail (p99) misses budgets long before the median; thermal throttling lives there |
| Throughput | Server cost per million requests | Batched, pipelined, many concurrent streams | Irrelevant to a batch-1 on-device feature; differs from latency by 10×+ |
| Parameters | Model capacity | Count from architecture | Says nothing about disk or speed until you fix a data type |
| Model size (MB) | App download / flash budget | `ls -l` the `.tflite`/`.gguf` | Params ≠ bytes; depends entirely on bits-per-weight |
| Peak memory | Will it OOM the device? | `max_memory_allocated`; declared tensor arena on MCU | Peak *activation* usually exceeds params; the cliff, not the floor |
| FLOPs / MACs | Theoretical work; capacity proxy | `thop` / `fvcore` / `ptflops` (mind ×2) | **Does not predict latency for memory-bound layers** |
| Energy / power | Battery life; thermal ceiling | Power monitor / `tegrastats` / `powermetrics`; subtract idle | Tracks bytes moved, not FLOPs; the radio dwarfs everything |

Now the fp32 → int8 before→after, on a named target, which is the result shape you'll produce after almost every quantization pass in this series. Numbers are representative for a MobileNetV2-class model on a phone NPU (Pixel-class), with the accuracy delta in the range typical of well-calibrated post-training quantization.

| Metric | fp32 baseline | int8 (PTQ) | Change |
| --- | --- | --- | --- |
| Top-1 accuracy | 72.0% | 71.4% | −0.6 pts |
| Model size on disk | ~14 MB | ~3.5 MB | **4× smaller** |
| Latency p50 (NPU) | ~9.0 ms | ~3.2 ms | **~2.8× faster** |
| Latency p99 (NPU) | ~14 ms | ~5.5 ms | ~2.5× faster |
| Peak working memory | ~24 MB | ~9 MB | ~2.7× less |
| Energy / inference | ~6.0 mJ | ~2.1 mJ | **~2.9× less** |

Two things to notice. First, every "good" metric moved together here — size, latency, memory, and energy all improved ~2.5–4× — because int8 quantization attacks the *bytes moved*, which is the common cause behind size, memory, and (per the Horowitz numbers) energy, and the int8 kernels are genuinely faster on an NPU built for them. Second, the latency gain (~2.8×) is *less* than the size gain (4×) — because the model wasn't purely memory-bound; the compute-bound layers got the math speedup but not the full 4× bandwidth relief, and some layers may have hit kernel or op-support limits. The size shrank exactly 4× (it's a pure bits-per-weight effect); latency shrank by a model-dependent amount governed by the roofline. *That gap between the size ratio and the latency ratio is itself a measurement that tells you how memory-bound your model is.* Reporting both is the honest move; reporting only the 4× size number oversells the user-facing speedup.

## Which metric should drive YOUR decision

There is no universal "most important metric" — there is the metric that maps to *your binding constraint*. Here is the decision logic, and it's worth being decisive about it because picking the wrong north-star metric is how you spend a sprint optimizing a number no one feels.

- **Real-time / interactive (camera filter, AR, wake word, live captions):** your north star is **p99 latency at batch 1** against a hard frame or response budget (16.7 ms for 60 fps, 33 ms for 30 fps, ~200 ms for "feels instant" taps). Optimize the *tail* and measure under *sustained thermal load*, not a cold burst. FLOPs are a distant proxy here; trust per-op on-device latency.
- **Battery-constrained / always-on (wearables, drones, IoT sensors, background features):** your north star is **energy per inference × duty cycle = mWh/hour**, and the dominant levers are "move fewer bytes" (quantization) and "don't transmit" (stay on-device, fuse to keep data on-chip). A model that's 10 ms faster but moves more bytes can be *worse* for battery.
- **Microcontroller / tiny-memory target (Cortex-M, RP2040, ESP32):** your north star is **peak activation memory against the SRAM budget** (often 256 KB–1 MB) and **model size against flash** — these are *hard, must-fit* constraints. If it doesn't fit, nothing else matters. Latency is usually a secondary concern once it fits.
- **App-size / distribution constrained (mobile app, OTA update, App Store size limits):** your north star is **model size on disk (MB)**, dominated by bits-per-weight. Quantization and weight-sharing/palettization are the direct levers; FLOPs are irrelevant to download size.
- **Server / batch backend (the cloud half of a hybrid product):** your north star is **throughput (inferences or tokens per second) per dollar**, where batching, sequence packing, and continuous batching dominate, and batch-1 latency matters only for the SLA tail.

The meta-rule: **pick exactly one north-star metric per deployment, derived from the binding constraint, and one or two guardrail metrics (usually accuracy, plus whichever of size/latency/memory/energy is the secondary risk). Then measure those, on the target, honestly. Everything else is diagnostic.** FLOPs and parameter count are *diagnostics* — useful for understanding *why* a number is what it is (they feed the roofline) — but they are almost never the right north star, because no user, battery, or chip directly experiences a FLOP count.

A subtlety for hybrid products that split work between device and cloud: the constraint can *change which half you're optimizing*. If a feature does cheap on-device pre-filtering and sends only hard cases to a server, then the on-device half is energy-and-latency constrained (it runs constantly) while the server half is throughput-and-cost constrained (it runs rarely but on big batches). The same model family deployed to both halves needs *two different north-star metrics*, measured on *two different targets*, and a FLOP count that looks fine for the server can be a battery disaster on the device. Resist the urge to find one number that scores both; the binding constraint is local to each deployment, and the honest answer is two measurements, not one average.

One more guardrail worth naming explicitly: **accuracy is almost always a guardrail, never a north star, in an optimization pass** — you fix a minimum acceptable accuracy and then push the efficiency metric as far as it goes while staying above that floor. The mistake is treating accuracy as something to maximize during deployment optimization; you already trained for accuracy, and now you're spending it deliberately to buy speed, size, or battery. The whole accuracy–efficiency Pareto frame this series is built on assumes you've decided how much accuracy you're *willing to spend*, then you measure what efficiency that buys. Without a stated accuracy floor, every efficiency number is unanchored — a 4× speedup means nothing if you don't say "at −0.6 points of accuracy."

## Case studies: the literature agrees, FLOPs lie

A few named results that make the point with real measurements, not toy numbers.

**ShuffleNet v2 (Ma et al., 2018), "Practical Guidelines for Efficient CNN Architecture Design."** The paper's whole thesis is that FLOPs are an *indirect* metric that mispredicts latency, and it demonstrates this directly on an ARM phone and an NVIDIA GPU. It shows that networks with *equal FLOPs* can have very different speeds depending on memory-access cost (MAC), degree of fragmentation (many small ops hurt parallelism), and element-wise op overhead — and it derives design guidelines (equal channel widths minimize memory access, avoid excessive group convolution, reduce fragmentation) that optimize *measured* speed, not FLOPs. If you read one paper after this post, read that one; it is the original "measure the right thing" manifesto for efficient CNNs.

**MobileNetV3 (Howard et al., 2019).** The architecture was found partly by hardware-aware NAS that optimized for *measured latency on a Pixel phone*, not FLOPs — and the resulting network sometimes has *more* FLOPs than a competitor while being faster on the actual device, precisely because the search rewarded layers the phone's runtime executes efficiently. The "h-swish" activation choice was driven by what was cheap to compute on the target, not by FLOP accounting.

**MCUNet (Lin et al., 2020).** On microcontrollers the binding metric is peak SRAM, and MCUNet co-designs the network (TinyNAS) and the inference engine (TinyEngine) to fit a model into ~320 KB of SRAM on a Cortex-M, running ImageNet-scale classification on hardware where the *activation memory*, not FLOPs, is the wall. It's the cleanest demonstration that "what fits" (peak memory) is a different and harder constraint than "how much math."

**MLPerf Inference / MLPerf Tiny benchmarks.** The industry-standard benchmarks deliberately report *measured* latency and energy on *named* hardware under controlled conditions (with rules about warm-up, percentiles, and power measurement), and notably do *not* let you submit FLOP counts as a result. The existence and design of MLPerf is itself an argument that the field decided FLOPs are not a credible efficiency metric — you have to measure on the target. MLPerf Tiny even has a standardized energy-measurement methodology because, on tiny devices, joules are the result that matters.

The throughline across all four: every serious efficiency result in the field is reported as *measured latency/energy/memory on named hardware*, with FLOPs relegated to a diagnostic. The literature already learned the lesson this post is teaching; the trap persists only because FLOPs are so much easier to compute.

## When FLOPs are actually fine (and when each metric breaks down)

In fairness, FLOPs are not useless — they're a *bad north star* but a *fine diagnostic* in specific cases, and being precise about this keeps you from overcorrecting.

- **FLOPs are a reasonable latency proxy when your model is genuinely compute-bound on the target** — big dense matmuls, large-channel standard convs, transformer prefill on a chip whose ridge point is below your layers' intensity. In that regime, $I > I^*$ for most layers, so $\text{time} \approx \text{FLOPs}/\pi$ and a FLOP cut really does cut time. The roofline tells you exactly when you're in this regime.
- **FLOPs are a fine *capacity* proxy** for comparing model families at the design stage before you have hardware, as a first-order filter. Just don't let it survive contact with a device.
- **Parameter count is a fine proxy for download size only after you fix the data type** — and a fine proxy for capacity/overfitting risk independent of hardware.

And each *other* metric has its own breakdown mode you should know:

- **Latency breaks down** if you don't fix the *conditions*: thermal state, clock governor, thread count, batch size, and warm-up all swing it 2–3×. A latency number without those conditions stated is not a measurement, it's a vibe.
- **Peak memory breaks down** across runtimes: the same model has different peaks under TFLite vs ONNX Runtime vs raw PyTorch because each does different memory planning and buffer reuse. Measure on the runtime you'll ship.
- **Energy breaks down** if you forget to subtract idle baseline, or measure the wrong rail, or change the duty cycle. "0.5 J per inference" is meaningless without "active minus idle, package rail, radio off, 30 fps sustained."

The discipline is the same for all of them: **state the conditions, measure on the target, report the distribution not a point, and subtract baselines.** A number without conditions is the FLOP-count mistake wearing a different hat.

## Key takeaways

- **FLOPs are not latency.** A model with half the FLOPs can run slower because the layers you cut were memory-bound, not compute-bound. Measure latency on the target before you believe a FLOP reduction.
- **Arithmetic intensity (FLOPs per byte) decides everything.** Above the chip's ridge point you're compute-bound and FLOPs predict time; below it you're memory-bound and FLOPs lie. Depthwise convs, element-wise ops, and LLM decode are the classic low-intensity, memory-bound traps.
- **Latency means batch-1 p99, not batched throughput and not the median.** The tail is where thermal throttling and the scheduler live, and it misses real-time budgets first. Throughput is a server metric; don't quote it for an on-device feature.
- **Peak activation memory, not parameter count, is what OOMs a device.** On an MCU the binding constraint is the activation set fitting in SRAM; a 200 KB model with a 700 KB activation peak simply won't run.
- **Energy tracks bytes moved, not arithmetic.** A DRAM access costs ~1000× a multiply, and the radio dwarfs DRAM. Quantization saves energy mainly by moving fewer bytes; staying on-chip and not transmitting are the biggest levers.
- **Measure honestly: warm up, lock or state clocks, synchronize the device, run many times, report the median plus p90/p99, subtract idle baselines.** Cold first calls, missing syncs, and reported means are the four ways your benchmark lies to you.
- **Pick one north-star metric from your binding constraint** — p99 latency for real-time, mWh/hour for battery, peak SRAM for MCUs, MB for app size, throughput-per-dollar for servers — plus accuracy as a guardrail. FLOPs and params are diagnostics, never the north star.
- **Report size *and* latency ratios after quantization.** Size shrinks exactly with bits-per-weight (a 4× int8 win is guaranteed); latency shrinks by a model-dependent, roofline-governed amount. The gap between them measures how memory-bound you are.

## Further reading

- **Williams, Waterman, Patterson (2009), "Roofline: An Insightful Visual Performance Model for Multicore Architectures"** — the original roofline paper; the source of the compute-bound vs memory-bound framing and the $\min(\pi, \beta \cdot I)$ ceiling. The mathematical backbone of this whole post.
- **Ma, Zhang, Zheng, Sun (2018), "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"** (ECCV) — the "FLOPs ≠ latency" manifesto, with direct on-device measurements and four design guidelines based on memory-access cost and fragmentation rather than FLOPs.
- **Horowitz (2014), "Computing's Energy Problem (and What We Can Do About It)"** (ISSCC keynote) — the energy-per-operation numbers showing DRAM access costs hundreds to thousands of times more than arithmetic; the basis for the energy-tracks-bytes argument.
- **MLPerf Inference and MLPerf Tiny benchmarks (MLCommons)** — the industry-standard, measured-on-named-hardware methodology for latency and energy, with rules about warm-up, percentiles, and power measurement. Read the rules to see how the field measures honestly.
- **Lin et al. (2020), "MCUNet: Tiny Deep Learning on IoT Devices"** (NeurIPS) — the cleanest case study of peak-SRAM-as-the-binding-constraint and hardware-aware co-design under a memory budget.
- Within this series: start from [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame, go deep on the bottleneck model in [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and see all the metrics applied end-to-end in [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
