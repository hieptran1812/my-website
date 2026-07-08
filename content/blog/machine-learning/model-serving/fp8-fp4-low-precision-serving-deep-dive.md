---
title: "FP8 and FP4 for LLM serving: the numerics of low-precision inference"
date: "2026-07-07"
publishDate: "2026-07-07"
description: "Go one level below the quantization playbook: what FP8 and FP4 actually are, how block scaling keeps accuracy, and how to serve them for real throughput on Hopper and Blackwell."
tags:
  [
    "model-serving",
    "inference",
    "fp8",
    "fp4",
    "quantization",
    "low-precision",
    "mxfp4",
    "nvfp4",
    "llm-inference",
    "gpu-optimization",
    "vllm",
    "tensor-cores",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-1.webp"
---

You have eight H100s serving Llama-3-70B in BF16. The model weights eat 140 GB, so each GPU carries a shard, and after weights and the runtime there is barely enough headroom for a healthy KV cache. Decode throughput sits around 2,600 tokens per second aggregate, GPU compute utilization hovers at 40% during decode, and the afternoon traffic wave is pushing p99 time-per-output-token past your 40 ms SLA. Someone in the channel says "just quantize it to FP8." You flip on `--quantization fp8`, the model now fits in 70 GB with room to double the KV cache, decode throughput jumps to roughly 4,300 tokens per second, and — this is the part that decides whether you keep the change — the eval suite barely moves. Perplexity drifts by a few hundredths, MMLU is within half a point, and nobody files a bug about garbled output.

That "barely moves" is not luck, and it is not a marketing claim. It is a property of the number format. FP8 E4M3 keeps three mantissa bits and a four-bit exponent, which is enough to represent transformer weights and activations to a relative error small enough that the model's next-token distribution barely shifts. FP4, on the other hand, keeps a single mantissa bit — it is a genuinely lossy format that only works because a shared block scale carries the magnitude for you. Whether the accuracy holds is decided by the format's bit layout and by how finely you scale, not by which flag you typed. This post is about those numerics: the layer directly beneath the [quantization-for-LLM-serving playbook](/blog/machine-learning/model-serving/quantization-for-llm-serving), where you stop treating "FP8" as a checkbox and start reasoning about exponents, mantissas, block scales, and where each one costs you.

By the end you will be able to: read a float format off its `E<x>M<y>` name and predict its range and precision; explain why activations need finer scaling than weights and what per-token, per-channel, and per-block scaling actually cost; pick a `WxAy` recipe (W8A8, W4A16, W4A4) for weights, activations, and the KV cache; quantize a model to FP8 with `llm-compressor` and serve it in vLLM with FP8 KV; and measure the accuracy damage with a logprob and KL-divergence harness before you ship. Everything here is a trade on the same serving triangle the rest of the series returns to — **latency, throughput, cost** — with a fourth axis, accuracy, that low precision spends deliberately.

![Matrix comparing FP16, FP8 E4M3, FP8 E5M2, FP4 MXFP4, and NVFP4 across bit layout, maximum magnitude, mantissa bits, required tensor-core hardware, and primary serving use](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-1.webp)

The figure above is the whole landscape on one page, and it is worth internalizing before we derive anything. Read it as a frontier: FP16 is the reference, FP8 E4M3 is the near-lossless serving default that needs Hopper-class tensor cores, and FP4 doubles the win again but only pays off on Blackwell and only when you have an accuracy budget to spend. No single format wins every column, which is exactly why format choice is an engineering decision and not a default.

## Why fewer bits is the deepest lever left

Every other serving optimization moves work around: continuous batching packs more requests into a step, paged attention stops fragmenting KV memory, speculative decoding guesses ahead. Precision is different — it changes the *unit cost* of the arithmetic itself. That gives it two separate payoffs, and confusing them is the most common mistake I see.

The first payoff is memory. LLM decode is memory-bandwidth bound: at the batch sizes you actually run, every decode step reads the entire weight matrix out of HBM to produce one token per sequence, and the arithmetic intensity is far below the hardware's compute-to-bandwidth ratio. I derive this in full in the [why-LLM-serving-is-different post](/blog/machine-learning/model-serving/why-llm-serving-is-different); the one-line version is that decode tokens per second is proportional to how fast you can stream weight bytes. Halve the bytes per parameter and you roughly halve the time per decode step. FP16 is 2 bytes per parameter, FP8 is 1 byte, FP4 is half a byte plus a small block-scale overhead. A 70B model is 140 GB in FP16, 70 GB in FP8, and about 37 GB in MXFP4.

Put concrete bandwidth on it. An H100 SXM has 3.35 TB/s of HBM3 bandwidth. Streaming 140 GB of BF16 weights takes about 42 ms, which caps a single decode stream near 24 tokens per second no matter how many FLOPs the chip can do. In FP8 the same weights are 70 GB and stream in 21 ms; in FP4 they are 37 GB and stream in about 11 ms. Those are ceilings, not measured throughput — real decode is lower because of attention, the KV cache read, and kernel overhead — but the *ratio* is what survives to production: cutting bytes per parameter cuts the memory-bound floor proportionally.

The ceiling scales with the chip, and the arithmetic is the same everywhere: single-stream tokens per second is at most ${\text{bandwidth} / \text{weight bytes}}$. Run it across three generations for the 70B model to see that precision and silicon are two multipliers on the same floor:

| GPU | HBM bandwidth | 70B in BF16 → ceiling | 70B in FP8 → ceiling | 70B in FP4 → ceiling |
|---|---|---|---|---|
| H100 SXM | 3.35 TB/s | 140 GB → ~24 tok/s | 70 GB → ~48 tok/s | 37 GB → ~90 tok/s |
| H200 | 4.8 TB/s | 140 GB → ~34 tok/s | 70 GB → ~69 tok/s | 37 GB → ~130 tok/s |
| B200 | 8.0 TB/s | 140 GB → ~57 tok/s | 70 GB → ~114 tok/s | 37 GB → ~216 tok/s |

These are per-stream upper bounds, and production runs well below them because a real step also reads the KV cache and pays kernel overhead — but the table is the reason the two levers stack. A B200 in FP4 has roughly ${8.0/3.35 \times 140/37 \approx 9\times}$ the single-stream memory ceiling of an H100 in BF16, before you have batched a single extra request. Most serving does batch, which trades this single-stream latency for aggregate throughput, but the memory a lighter format frees is exactly what lets you batch harder.

The second payoff is compute, and this is where FP8 and FP4 differ from weight-only integer quantization. When both operands of a matrix multiply are low precision, the GPU dispatches a low-precision tensor-core kernel that is genuinely faster per FLOP. The rule of thumb is that each halving of precision doubles peak tensor-core throughput: on an H100, BF16 matrix math peaks near 990 dense TFLOPS while FP8 peaks near 1,980; on a B200, FP8 peaks near 4.5 dense PFLOPS while FP4 peaks near 9. (Vendors often quote the sparse numbers, which are 2x larger again; use the dense figures for capacity planning and treat all of them as peak, not sustained.) This compute win only materializes if the activations are also quantized — a W4A16 scheme that keeps activations in FP16 still runs an FP16 GEMM and gets no tensor-core speedup, only the memory win. A W8A8 scheme in FP8 gets both.

![Dataflow graph showing FP16 weights and FP16 activations quantized on separate paths into FP8, meeting at one FP8 tensor-core GEMM, accumulating in FP32, then dequantized once at the end](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-7.webp)

The dataflow above is the mechanical heart of a quantized linear layer, and it explains the compute win precisely. Weights are quantized once, offline, with a per-channel scale ${s_w}$. Activations are quantized on the fly, per token, with a scale ${s_x}$ computed from that token's own values. Both FP8 operands feed a single tensor-core GEMM that runs at the FP8 rate. Crucially, the products accumulate in FP32 — the accumulator never sees the precision loss, only the inputs do — and there is exactly one dequantization at the very end, where the FP32 result is multiplied by ${s_w \cdot s_x}$ to recover magnitude. This is why FP8 is nearly free: the expensive inner loop runs twice as fast, and the only lossy operations are the two input quantizations, which touch values the model was never that sensitive to.

The prefill phase, which processes the whole prompt in parallel, is compute-bound rather than memory-bound, so it benefits from the tensor-core win directly. That makes FP8 a rare optimization that helps both halves of LLM inference: prefill gets faster GEMMs, decode gets less bandwidth pressure. If you want to see where each phase sits on the compute-versus-bandwidth line, the [roofline-analysis post](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference) walks through the arithmetic-intensity plot that makes this obvious.

## The float formats, one bit at a time

A floating-point number is three fields: a sign bit, an exponent that sets the magnitude, and a mantissa (the fraction) that sets the precision within that magnitude. The value is roughly ${\pm 1.\text{mantissa} \times 2^{\text{exponent} - \text{bias}}}$. Everything about a format's behavior falls out of how many bits go to the exponent versus the mantissa. More exponent bits buy dynamic range — the ratio between the largest and smallest representable magnitude. More mantissa bits buy relative precision — how finely you can distinguish two nearby values. You are always trading one against the other, and when you shrink the total bit budget you have to decide which to sacrifice.

![Grid showing the bit layout of FP16, FP8 E4M3, FP8 E5M2, and FP4 E2M1, with the sign, exponent, and mantissa fields shrinking as total width drops and a block-scale annotation for FP4](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-2.webp)

Walk the figure top to bottom. FP16 is 1 sign, 5 exponent, 10 mantissa bits: 10 mantissa bits give a relative precision near ${2^{-11}}$, fine enough that FP16 is treated as ground truth for serving. The two FP8 formats split the byte differently. **E4M3** spends 4 bits on the exponent and 3 on the mantissa. Its maximum magnitude is 448, its smallest normal value is ${2^{-6}}$, and it reaches down to ${2^{-9}}$ with subnormals. It has no infinities — the standard reclaims that encoding to push the max from the naive 240 up to ${1.75 \times 2^8 = 448}$ — and only two NaN patterns. Three mantissa bits give a relative precision near ${2^{-4}}$, or about 6%, per element. **E5M2** spends 5 bits on the exponent and 2 on the mantissa: its max magnitude is ${1.75 \times 2^{15} = 57344}$ and it keeps IEEE-style infinities and NaNs, but with only 2 mantissa bits its relative precision is near ${2^{-3}}$, about 12%.

That difference decides where each format goes. Inference weights and activations, once you have subtracted the mean and handled outliers with scaling, live in a fairly narrow band of magnitudes — you do not need a range of 57,344, you need precision. So **E4M3 is the inference default for both weights and activations**: the extra mantissa bit matters more than the extra range. E5M2's wider range earns its keep in training, where gradients span many orders of magnitude and an overflow to infinity is worse than a coarse value; the mixed-precision training recipes keep gradients in E5M2 and the forward pass in E4M3 for exactly this reason. For serving, you will almost always want E4M3, and when a tool says "FP8" without qualification it usually means E4M3.

One nuance worth carrying: BF16, the other common serving baseline, is itself a range-over-precision trade against FP16. BF16 keeps 8 exponent bits and 7 mantissa bits, giving it the same enormous range as FP32 but only 7 mantissa bits of precision. When your baseline is BF16 rather than FP16, the mantissa gap down to E4M3 is 7 bits to 3 bits, not 10 to 3 — the drop is smaller than the format name suggests, which is part of why FP8 lands so softly on models trained in BF16.

| Format | S / E / M | Max magnitude | Min normal | Relative precision | Tensor cores | Serving role |
|---|---|---|---|---|---|---|
| FP16 | 1 / 5 / 10 | ±65504 | ${2^{-14}}$ | ~${2^{-11}}$ | all GPUs | baseline reference |
| BF16 | 1 / 8 / 7 | ±3.4e38 | ${2^{-126}}$ | ~${2^{-8}}$ | all GPUs | baseline reference |
| FP8 E4M3 | 1 / 4 / 3 | ±448 | ${2^{-6}}$ | ~${2^{-4}}$ | Hopper, Ada+ | weights + activations |
| FP8 E5M2 | 1 / 5 / 2 | ±57344 | ${2^{-14}}$ | ~${2^{-3}}$ | Hopper, Ada+ | wide range, gradients |
| FP4 E2M1 | 1 / 2 / 1 | ±6 (× scale) | ${2^{-1}}$ | ~${2^{-2}}$ | Blackwell+ | weights (+ activations) |

FP4 is the bottom row and it is a different animal. E2M1 — 1 sign, 2 exponent, 1 mantissa bit — represents exactly sixteen values: zero, plus and minus {0.5, 1, 1.5, 2, 3, 4, 6}. Its maximum magnitude on its own is 6, and its relative precision is a brutal ${2^{-2}}$, roughly 25% between adjacent values in the worst case. No transformer survives being cast into a set of sixteen numbers directly. FP4 is only usable because it is never used alone: it always comes with a shared block scale, which is the subject of microscaling and the next two sections.

#### Worked example: how E4M3 reaches ±448, and what an element costs

The ±448 maximum is not a round number someone picked; it falls straight out of the bit layout, and working it by hand fixes where E4M3's precision actually lives. E4M3 uses an exponent bias of 7, so the stored exponent field ${e \in \{1,\dots,15\}}$ maps to an unbiased exponent ${e-7 \in \{-6,\dots,+8\}}$. The largest finite value uses the top exponent field 1111 (unbiased ${+8}$) with mantissa ${110_2 = 1.75}$ — the single pattern ${1.111_2}$ at that exponent is reserved for NaN, which is why the max is ${1.75 \times 2^{8} = 448}$ rather than the naive ${1.875 \times 2^{8} = 480}$. The smallest normal value is ${1.0 \times 2^{-6}}$, and subnormals reach down to ${(1/8) \times 2^{-6} = 2^{-9}}$. There are no infinities: reclaiming those encodings is exactly the trick that buys the wider finite range.

Now quantize a concrete weight, ${w = 0.34}$, to both FP8 formats and watch the mantissa bits do their job. Write ${0.34 = 1.36 \times 2^{-2}}$. In E4M3 the three mantissa bits give the grid ${\{1.000, 1.001, \dots, 1.111\}}$ in steps of ${1/8 = 0.125}$; the value ${1.36}$ lands between ${1.010_2 = 1.25}$ and ${1.011_2 = 1.375}$ and rounds to ${1.375}$, so ${\hat{w} = 1.375 \times 2^{-2} = 0.34375}$ — a relative error of ${1.1\%}$, and the worst case anywhere in that binade is half a step, ${0.5 \times 2^{-2} \times 2^{-3} \approx 4.6\%}$. In E5M2 the two mantissa bits give a coarser grid in steps of ${0.25}$; ${1.36}$ rounds down to ${1.01_2 = 1.25}$, so ${\hat{w} = 1.25 \times 2^{-2} = 0.3125}$ — a relative error of ${8\%}$. Same number, same 8-bit budget, and the one extra mantissa bit roughly halved the error. Multiply that gap across a hundred billion weights and you have the whole reason E4M3, not E5M2, is the serving format: inference has already scaled its tensors into a narrow band, so it spends its bits on precision, not on a range it will never use.

## Scaling: why one number can't cover a tensor

Quantization to a fixed-point or narrow-float grid is always the same operation. Pick a scale ${s}$, divide, round to the nearest representable value, and store the integer or narrow-float code. To read the value back, multiply by the scale again:

$$x_q = \operatorname{round}\!\left(\frac{x}{s}\right), \qquad \hat{x} = s \cdot x_q$$

The scale ${s}$ maps the real range you care about onto the representable range of the format. For a ${b}$-bit signed integer covering ${[-R, R]}$ the scale is ${s = R / (2^{b-1}-1)}$; for a narrow float the format's own exponent handles most of the range and the scale mostly recenters and prevents overflow past the format max. Either way the rounding introduces an error bounded by half a step: ${|x - \hat{x}| \le s/2}$. That bound is the entire story. Everything about scaling granularity is a fight to make ${s}$ small where it matters.

Here is the tension. To avoid clipping, ${s}$ has to be large enough that the largest value in whatever region shares that scale still fits under the format's max. If a single tensor shares one scale — per-tensor quantization — then ${s}$ is set by the largest element anywhere in that tensor. Transformer activations are famous for having rare, enormous outliers: specific channels, in specific layers, that spike to values 50–100x the typical magnitude. One such outlier forces the whole tensor's scale up, and now every ordinary value is quantized with a huge step size and loses most of its precision.

![Before-and-after comparison: per-tensor scaling forced up by an outlier of magnitude 340 crushes normal values, versus per-block scaling that isolates the outlier so ordinary values keep their precision](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-3.webp)

The figure makes the failure concrete with numbers. Say an activation tensor has one outlier at magnitude 340 while almost every other value sits near 2. With per-tensor INT8 scaling, ${s = 340/127 \approx 2.7}$, so a normal value of 2 rounds to ${\operatorname{round}(2/2.7) = 1}$ — it collapses onto a single code, and values between roughly 1.3 and 4.0 all map to the same integer. The tensor's real information lived in those normal values, and the scale just threw most of it away to make room for one outlier. Per-block scaling breaks the tensor into small groups and gives each group its own scale set by that group's local maximum. The block containing the outlier gets a large scale; every other block gets a scale near ${2/127}$ and keeps the normal values crisp. The outlier is quarantined, and the error it would have imposed on the whole tensor is confined to its own block.

This is *why activations need finer scaling than weights.* Weights are trained with regularization and weight decay; their distribution per output channel is well-behaved and a single scale per channel captures it. Activations are the running state of the model and carry these outlier channels, so they need per-token or per-block scaling to stay accurate. The [SmoothQuant](/blog/machine-learning/model-serving/quantization-for-llm-serving) insight — which I will come back to under calibration — is a clever dodge: mathematically migrate the outlier magnitude out of the activations and into the weights, so both end up easy to quantize with coarse scales. But absent that trick, the rule stands: the harder a tensor's distribution, the finer the scale it needs.

## Scaling granularity: the accuracy knob

Granularity is a spectrum, and moving along it trades accuracy against overhead. At the coarse end, one scale per tensor costs almost nothing to store or compute but clips outliers. At the fine end, a scale per small block of elements is robust to outliers but stores and computes many more scales. The right point depends on whether you are scaling weights or activations and on how many bits the elements have — fewer element bits demand finer scaling to compensate.

![Matrix comparing per-tensor, per-channel, per-token, per-block microscaling, and per-group scaling across scale count, overhead, outlier robustness, and typical use](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-5.webp)

The five rows in the figure are the ones you will meet in practice. **Per-tensor**: one scale, cheapest, used for weights when the model is forgiving and for the simplest FP8 paths. **Per-channel**: one scale per output channel of a weight matrix, computed offline, essentially free at inference and the standard for weight quantization. **Per-token**: one scale per row of the activation matrix, computed dynamically at runtime from that token's values — this is the workhorse for activation quantization because it costs one reduction per token and handles the fact that different tokens have different magnitudes. **Per-block (microscaling)**: one scale per fixed block of elements, typically 32, and the mechanism that makes FP4 viable. **Per-group**: one scale per group of weights along the input dimension, typically 128, which is what AWQ and GPTQ use for 4-bit integer weights.

The formula that makes microscaling precise is worth writing out, because it is the difference between FP4 that works and FP4 that produces noise. For a block of ${k}$ elements ${\{x_i\}}$ with element format maximum exponent ${e_{\max}}$, the shared scale is chosen to align the block's largest value with the top of the element range:

$$X = 2^{\left\lfloor \log_2\left(\max_i |x_i|\right)\right\rfloor - e_{\max}}, \qquad q_i = \operatorname{RN}\!\left(\frac{x_i}{X}\right), \qquad \hat{x}_i = X \cdot q_i$$

For MXFP4, ${k = 32}$, the element format is E2M1 (so ${e_{\max} = 2}$, since the max value 6 equals ${1.5 \times 2^2}$), and the shared scale ${X}$ is stored as an 8-bit power of two — the E8M0 format, which is pure exponent with no mantissa. The storage cost is 4 bits per element plus 8 bits per 32 elements, which is ${4 + 8/32 = 4.25}$ bits per element on average, about 0.53 bytes per parameter. **NVFP4** refines this in two ways: it uses a smaller block of ${k = 16}$ so each scale covers fewer elements, and it stores the block scale as an FP8 E4M3 value — a scale *with* mantissa bits, so the scale itself is finer than a bare power of two — plus a single FP32 per-tensor scale on top. That two-level scaling is why NVFP4 recovers a meaningful fraction of the accuracy that MXFP4 leaves on the table.

Now the derivation that makes all of this rigorous — *why finer granularity bounds the error.* Take the relative error of one element. Its absolute error is bounded by half a step, ${|x_i - \hat{x}_i| \le s_{\text{local}}/2}$, where ${s_{\text{local}}}$ is the effective step of the block that contains it. Under per-block scaling, ${s_{\text{local}}}$ is proportional to that block's local maximum, ${\max_{j \in \text{block}} |x_j|}$. So the *relative* error of a typical element is bounded by

$$\frac{|x_i - \hat{x}_i|}{|x_i|} \;\lesssim\; \frac{1}{2} \cdot \frac{s_{\text{local}}}{|x_i|} \;\propto\; \frac{\max_{j \in \text{block}} |x_j|}{|x_i|} \cdot 2^{-(m+1)}$$

where ${m}$ is the element mantissa bit count. Under per-tensor scaling the numerator is the *global* maximum, which for an outlier-heavy tensor can be 50–100x larger than any given block's local maximum. Shrinking the region each scale covers shrinks the ratio ${\max_{\text{block}}/|x_i|}$ toward 1, which drives the relative error toward the format's intrinsic floor of ${2^{-(m+1)}}$ regardless of the element's magnitude. That is the precise sense in which finer scaling "bounds the error": it makes the achievable relative precision uniform across the tensor instead of hostage to its single worst element. And it tells you the cost knob directly — the number of scales is ${(\text{tensor size}) / k}$, so halving the block size ${k}$ doubles the scale storage and the reduction work. FP4 pays that overhead because its 1 mantissa bit needs every bit of help the scale can give; FP8's 3 mantissa bits are forgiving enough that per-tensor or per-channel scaling is usually plenty.

#### Worked example: microscaling a 32-value block into MXFP4

Nothing makes microscaling click like pushing an actual block through it. Take a block of weights whose largest magnitude is ${3.0}$. The shared scale aligns that maximum with the top of the E2M1 range: ${X = 2^{\lfloor \log_2 3.0 \rfloor - e_{\max}} = 2^{1 - 2} = 2^{-1} = 0.5}$, stored as the bare 8-bit exponent ${-1}$ in E8M0. Every element is divided by ${0.5}$, rounded to the nearest of the sixteen E2M1 codes, and stored as 4 bits. Reconstruction multiplies back by ${0.5}$:

| Original ${x_i}$ | ${x_i / X}$ | Nearest E2M1 ${q_i}$ | ${\hat{x}_i = 0.5\,q_i}$ | Error |
|---|---|---|---|---|
| ${3.00}$ | ${6.00}$ | ${6}$ | ${3.00}$ | ${0.00}$ |
| ${-2.10}$ | ${-4.20}$ | ${-4}$ | ${-2.00}$ | ${0.10}$ |
| ${1.40}$ | ${2.80}$ | ${3}$ | ${1.50}$ | ${0.10}$ |
| ${0.90}$ | ${1.80}$ | ${2}$ | ${1.00}$ | ${0.10}$ |
| ${0.05}$ | ${0.10}$ | ${0}$ | ${0.00}$ | ${0.05}$ |

The block max is exact, the mid-range values carry a step of error near ${0.1}$, and the tiny ${0.05}$ underflows to zero because E2M1's smallest nonzero code is ${0.5 \times 0.5 = 0.25}$ after scaling. That underflow is the honest cost of 4 bits, and it is precisely what NVFP4 attacks by covering only 16 elements per scale (so a small value is less likely to share a block with a large one) and by storing the scale in FP8 E4M3 rather than a bare power of two, so ${X}$ itself can be a value like ${0.47}$ instead of being snapped to ${0.5}$. Here is the whole mechanism in a few lines of PyTorch, which is exactly what a Blackwell tensor core does in silicon:

```python
import torch

# The 16 signed E2M1 codes: {0, 0.5, 1, 1.5, 2, 3, 4, 6} and their negatives.
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
E2M1 = torch.cat([-E2M1.flip(0), E2M1[1:]])  # symmetric grid, drop the duplicate 0

def to_nearest(x, grid):
    # Snap each value to the closest representable code (nearest-neighbour on the grid).
    idx = (x.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return grid[idx]

def mxfp4_block(x, block=32, e_max=2):
    # One E8M0 (power-of-two) scale per block, chosen to align the block max
    # with the top of the E2M1 range, then quantize + dequantize.
    x = x.reshape(-1, block)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    exp = torch.floor(torch.log2(amax)) - e_max      # E8M0 exponent per block
    scale = torch.exp2(exp)                            # the shared block scale X
    q = to_nearest(x / scale, E2M1)                    # 4-bit codes
    return (q * scale).reshape(-1)                     # dequantized values

torch.manual_seed(0)
w = torch.randn(4096) * 0.7                            # a well-behaved weight vector
w_hat = mxfp4_block(w)
print(f"MXFP4 mean rel err: {((w - w_hat).abs() / w.abs().clamp_min(1e-3)).mean():.4f}")
print(f"bits/element: {4 + 8/32}")                     # 4.25, i.e. ~0.53 bytes/param
```

Swap `block=32` for `block=16` and store `scale` in E4M3 and you have moved from MXFP4 to the NVFP4 recipe; the error drops and the bits-per-element rises to ${4 + 8/16 = 4.5}$. That single knob — how many elements share one scale, and how precisely that scale is stored — is the entire design space between "4-bit that produces noise" and "4-bit that ships."

## What you actually quantize: weights, activations, and the KV cache

"Quantizing the model" is three separable decisions, and naming them precisely is half the battle. The industry notation is `WxAy`: `W` is the weight bit-width, `A` is the activation bit-width. `W4A16` means 4-bit weights, 16-bit activations — weight-only, the memory win without the compute win. `W8A8` means 8-bit weights and 8-bit activations — both wins, the FP8 tensor-core path. `W4A4` means 4-bit everything — the aggressive FP4 regime. The KV cache is a third target the notation does not name, and it is often the cheapest win of the three.

| Recipe | Weight bits | Activation bits | GEMM runs at | Memory win | Where it fits |
|---|---|---|---|---|---|
| W16A16 | 16 | 16 | FP16/BF16 | baseline | reference |
| W8A8 | 8 | 8 | FP8 (fast) | 2x weights | FP8 serving default, Hopper/Ada |
| W4A16 | 4 | 16 | FP16 (no speedup) | 4x weights | memory-bound decode, Ampere, AWQ/GPTQ |
| W4A4 | 4 | 4 | FP4 (fast) | 4x weights | throughput-bound, Blackwell |
| +FP8 KV | — | — | — | 2x KV cache | orthogonal — composes with any row above |

The last row is the one worth staring at: FP8 KV is not a `W` or an `A`, it is a separate lever on the attention memory, and you can throw it on top of any weight/activation recipe — even a plain BF16 model gets a longer context for one flag. Read `W4A16` and `W8A8` as answering *different* questions. `W4A16` asks "how do I fit the weights and stop starving the KV cache," and it answers with memory only because the FP16 activations force an FP16 GEMM. `W8A8` asks "how do I also make the matrix multiply itself faster," and it answers by quantizing both operands so a low-precision tensor-core kernel can run. `W4A4` is the greedy option that wants both wins at 4 bits and pays for it in accuracy. The figure lays the three physical targets out as a stack you compose from.

![Layered stack showing the three quantization targets — weights, activations, and the KV cache — with the WxAy notation and a composed serving recipe of W8A8 plus FP8 KV](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-4.webp)

The stack in the figure is the menu. **Weights** are the biggest memory line item and the easiest to quantize well, because you do it offline with full information about the distribution; W8 halves them, W4 quarters them. **Activations** are harder — they carry the outliers from the last section — but quantizing them is what unlocks the low-precision GEMM, so `A8` is the price of the compute win. **The KV cache** is the running memory of attention: for every token in every sequence, you store the key and value vectors so future tokens can attend to them. It grows linearly with batch size and context length and, at long context, it can dwarf the weights. Quantizing it to FP8 halves its footprint versus FP16, which directly buys you either longer context or more concurrent sequences in the same GPU memory — and its accuracy cost is small because attention is a weighted average that is naturally tolerant of noise in the values it averages. The [KV-cache-optimization post](/blog/machine-learning/model-serving/kv-cache-optimization) covers the paging and eviction mechanics; here the point is just that FP8 KV is a lever you throw independently of whether you quantized the weights.

The serving recipe you compose from these is the bottom of the stack: `W8A8` in FP8 for the compute-and-memory win on the linear layers, plus FP8 KV for the context win on attention. On Hopper that combination is close to a free lunch. Let me put real numbers on the KV cache piece, because it is the one people underestimate.

#### Worked example: FP8 KV cache on Llama-3-70B

Llama-3-70B has 80 layers and uses grouped-query attention with 8 key-value heads of dimension 128, so the per-token KV state is ${2 \times 80 \times 8 \times 128 = 163{,}840}$ values (the 2 is key plus value). In FP16 that is ${163{,}840 \times 2 = 327{,}680}$ bytes per token, or about 0.31 MB per token. At a context of 8,192 tokens, one sequence's KV cache is ${8192 \times 0.31 \approx 2.56}$ GB. On an 8xH100 node with the 70B weights taking 70 GB in FP8 (so under 9 GB per GPU shard) and roughly 60 GB free per GPU after the runtime, the KV cache is what limits concurrency.

Halve it with FP8 KV and each sequence's 8K-token cache drops to about 1.28 GB. Across the node, if KV memory was the binding constraint, you have just doubled the number of 8K-context sequences you can hold — which for a memory-bound decode workload roughly doubles the batch size and therefore the aggregate throughput. The accuracy cost of FP8 KV on this model is typically a perplexity drift in the low hundredths and no measurable task-accuracy change, because, again, attention averages over the values and washes out the per-element quantization noise. This is the single highest-leverage flag in the post: one `--kv-cache-dtype fp8` doubles your effective KV budget for a cost that hides inside eval noise.

One subtlety on the KV format itself: keys and values do not have to share a dtype with the weights, and the choice between E4M3 and E5M2 for the cache is a real one. E4M3 is the default because keys and values, once scaled, live in a narrow band and want the extra mantissa bit — but a handful of models push a few key channels to large magnitudes at very long context, and there E5M2's wider range avoids clipping at the cost of one mantissa bit. If a long-context model degrades under `fp8_e4m3` KV, trying `fp8_e5m2` is a cheaper first move than abandoning FP8 KV altogether.

## Outliers, and the layers that break low precision

Everything above assumed the tensor's distribution is the average case. The reason FP8 occasionally *does* hurt — and the reason FP4 needs so much scaffolding — is that transformer activations contain a small number of genuinely pathological values, and knowing where they live turns "quantization broke my model" from a mystery into a two-line fix.

The phenomenon has a name in the literature: **massive activations** (sometimes "outlier features"). In most layers a handful of specific hidden dimensions carry values one to two orders of magnitude larger than everything else, and they are remarkably consistent — the same channels spike across many tokens, and they concentrate in particular positions, often the very first token or delimiter tokens that the model uses as a kind of attention sink. These are not noise; the model relies on them, which is exactly why crushing them with a coarse scale hurts. Per-token and per-block scaling exist to quarantine them, but even fine scaling has limits when a single channel is 100x its neighbours and you only have four bits to spend.

They also concentrate in *particular layers*. Empirically the worst offenders are the FFN down-projection — the second linear in the MLP block, the one whose input is the post-activation hidden state that a GELU or SwiGLU can blow up — and, to a lesser extent, the attention output projection. The first few layers and the last few layers of the stack tend to be more sensitive than the middle. This geography is the practical lever: you do not have to quantize uniformly. Every serious toolchain lets you pass an `ignore` list — modules kept in BF16 while everything else goes to FP8 or FP4 — and the standard move is to keep `lm_head` high-precision (its errors land directly on the output logits) and, if a sensitivity sweep flags them, a small number of down-projection layers as well. Leaving 2–3% of the GEMM FLOPs in BF16 to protect the pathological layers costs almost nothing in speed and routinely recovers most of a surprising accuracy drop.

How do you find those layers without guessing? Run a per-layer sensitivity sweep: quantize one layer at a time, measure the KL divergence of the output against the full-precision model on a calibration set, and rank. The layers whose individual quantization moves the output distribution the most are the ones to protect. This is the same measurement discipline the final section builds into a ship gate, applied one layer deep — and it is the reason FP4 leans so hard on quantization-aware training. QAT lets the surviving high-precision layers *adapt* to the quantization noise their neighbours inject, which is the only way to recover accuracy once the outliers exceed what a block scale can absorb in four bits. FP8's three mantissa bits usually have enough headroom that the `ignore` list alone is sufficient; FP4 usually does not.

## Calibration: static, dynamic, and moving the difficulty around

A scale has to come from somewhere. There are two ways to get it, and the choice is a real latency-versus-accuracy trade.

**Dynamic scaling** computes the scale at runtime from the actual values being quantized. For activations, that means a per-token reduction — find the max magnitude of this token's row, derive the scale, quantize — on every forward pass. It adapts perfectly to whatever the input looks like, so it is the most accurate, and it needs no calibration data at all. Its cost is the reduction kernel and a little extra memory traffic per layer. For FP8 activations this overhead is small and dynamic per-token is the common default. **Static scaling** computes the scales once, offline, from a calibration pass over a small representative dataset, and bakes them into the checkpoint. At runtime there is no reduction — you just divide by the stored scale — so it is faster, but the scale is only as good as the calibration set's coverage. If production traffic has magnitudes the calibration set never saw, static scales clip.

The calibration set is smaller than people expect: a few hundred sequences of a few thousand tokens each, drawn from data that resembles production, is typically enough to estimate stable activation ranges. You are not training anything; you are just observing the distribution of activation magnitudes so you can set scales that do not clip. A common mistake is calibrating on the wrong domain — calibrating a code model on web text, say — which produces scales that clip on the real workload. Match the calibration data to the traffic.

The cleverest calibration methods do not just measure the difficulty; they *move* it. **SmoothQuant** observes that the outliers live in activations, which are hard to quantize, while weights are easy — so it multiplies a per-channel smoothing factor into the activations and divides it out of the weights, a mathematically equivalent transformation that shifts magnitude from the activation side to the weight side. After smoothing, both tensors have gentle distributions that quantize cleanly with coarse scales, and W8A8 becomes accurate. **AWQ** (activation-aware weight quantization) makes a related move for weight-only 4-bit: it uses a small calibration set to find the salient weight channels — the ones whose corresponding activations are large and therefore matter most to the output — and scales them up before quantization so they land on a finer part of the grid, protecting the channels the model is most sensitive to. Both are, at heart, ways of redistributing quantization difficulty so that a coarse, cheap scale can do a fine, expensive scale's job. They matter less for FP8, whose mantissa is generous, and a lot for 4-bit, where every bit of protection counts.

The SmoothQuant transform is worth making numeric, because the whole trick is one per-channel factor. For input channel ${j}$, the smoothing factor is ${s_j = \max(|X_j|)^{\alpha} / \max(|W_j|)^{1-\alpha}}$, where ${X_j}$ is the activation column and ${W_j}$ the corresponding weight row, and ${\alpha \in [0,1]}$ is a migration strength you tune (0.5 is a common start). You then run the layer as ${(X / s) \cdot (s \cdot W)}$, which is algebraically identical to ${X \cdot W}$ but moves magnitude from the activation onto the weight. Suppose a nasty channel has activation max ${340}$ and weight max ${0.5}$. With ${\alpha = 0.5}$, ${s_j = \sqrt{340}/\sqrt{0.5} = 18.4 / 0.707 = 26.1}$. The activation channel's max drops to ${340 / 26.1 = 13.0}$ and the weight channel's rises to ${0.5 \times 26.1 = 13.0}$ — the difficulty is now split evenly, and a ${340}$-to-${13}$ reduction on the activation side is the difference between a per-tensor scale that clips everything and one that does not. Turn ${\alpha}$ up and you push more onto the weights (good when weights are the easy side, which they usually are); turn it down and you keep more on the activations.

**GPTQ** attacks the weight-only problem from a different angle — it does not move difficulty, it *compensates* for it. Quantizing weights one column at a time introduces an error; GPTQ uses second-order (Hessian) information from a calibration set to update the not-yet-quantized weights so they absorb that error, minimizing the change in the layer's output rather than the change in the weights themselves. It is descended from Optimal Brain Quantization, and in practice it and AWQ are the two 4-bit weight-only workhorses you will see referenced: AWQ protects the salient channels, GPTQ corrects for the quantization it has already committed. For FP8 neither is usually necessary; for W4A16 one of them is essentially mandatory if you want to stay near the baseline.

One last calibration nuance that bites people: static activation scales interact badly with distribution shift. If you calibrate on English chat and then serve a burst of code or a new language, the activation magnitudes can exceed the calibrated range and clip. Dynamic per-token scaling has no such failure mode because it recomputes the scale from the live input, which is a large part of why FP8 serving defaults to dynamic activations despite the small runtime cost — the robustness is usually worth more than the reduction kernel it saves.

## The hardware reality: which chip runs which format

Low precision is only fast if the silicon has tensor cores for it. This is the constraint that most often decides your recipe, and it is worth being exact about.

FP8 tensor cores arrived with **Hopper** (H100, H200) and **Ada Lovelace** (L40S, L4, RTX 40-series) in 2022. So FP8 serving is available not just on the flagship datacenter parts but on the cheaper Ada inference cards, which is useful for cost-sensitive fleets. On **Ampere** (A100, A10) and older, there are no FP8 tensor cores: you can store weights in FP8 to save memory, but the GEMM runs by upcasting to FP16, so you get the memory win and none of the compute win. On Ampere, the low-precision compute win comes from INT8 tensor cores instead — which is why W8A8-INT8 (SmoothQuant) and W4A16-INT4 (AWQ, GPTQ) are the schemes that make sense there.

FP4 tensor cores arrived with **Blackwell** (B100, B200, GB200, and the RTX 50-series) in 2024–2025. Blackwell's tensor cores natively execute the MXFP4 and NVFP4 microscaled formats, applying the block scale in hardware, which is what makes FP4 fast rather than merely small. On any pre-Blackwell chip you can store MXFP4 weights but there is no native FP4 GEMM, so the format has to be upcast — you keep the memory win and lose the compute win, the same asymmetry as FP8 on Ampere. This is the single most important sentence in the post for planning: **FP8 needs Hopper or Ada, FP4 needs Blackwell**, and buying the format without the silicon buys you half the benefit.

| Precision | Tensor-core throughput vs BF16 | First supported | Notes |
|---|---|---|---|
| BF16 / FP16 | 1x (baseline) | Volta / Ampere | universal |
| FP8 (E4M3) | ~2x | Hopper, Ada | inference default |
| INT8 | ~2x | Turing / Ampere | Ampere's low-precision path |
| FP4 (MXFP4/NVFP4) | ~4x vs BF16 (~2x vs FP8) | Blackwell | microscale applied in HW |

The throughput column is the peak dense tensor-core rate relative to BF16 on the same generation; treat it as an upper bound on the speedup, since real kernels are limited by memory, overhead, and the un-quantized parts of the model. The [GPU-architecture-specific tuning post](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving) goes deeper on how these tensor-core rates interact with occupancy and kernel selection on each generation.

## Serving it: llm-compressor, vLLM, and the flags

Enough theory — here is the actual toolchain. The modern path is: produce a quantized checkpoint in the `compressed-tensors` format with `llm-compressor`, then serve it with vLLM. FP8 is the easy case because vLLM can also quantize on the fly at load time, so you can get started with a single flag.

The simplest possible FP8 serving, quantizing a BF16 checkpoint dynamically as vLLM loads it:

```bash
# Dynamic FP8: vLLM quantizes weights to FP8 E4M3 at load time and uses
# dynamic per-token FP8 activations. No calibration, no offline step.
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 8 \
  --max-model-len 8192
```

That gets you W8A8 FP8 with FP8 KV on an 8-GPU node. Dynamic load-time quantization is convenient but recomputes the weight scales every time you start the server and uses per-tensor weight scales; for production you usually want to quantize once, offline, with per-channel weight scales and (optionally) calibrated activation scales, then serve the resulting checkpoint. That is what `llm-compressor` produces:

```python
# Offline FP8 W8A8 with per-channel weight scales and dynamic activations.
# Produces a compressed-tensors checkpoint vLLM loads directly.
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "meta-llama/Llama-3.1-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# FP8_DYNAMIC: static per-channel weight scales, dynamic per-token activation
# scales. No calibration data needed because activations are scaled at runtime.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],  # keep the output projection in high precision
)

oneshot(model=model, recipe=recipe)
model.save_pretrained("Llama-3.1-70B-FP8-dynamic", save_compressed=True)
tokenizer.save_pretrained("Llama-3.1-70B-FP8-dynamic")
```

If you want *static* activation scales — faster at runtime, no per-token reduction — you switch the scheme to `FP8` (not `FP8_DYNAMIC`) and pass a calibration dataset so `llm-compressor` can observe activation ranges:

```python
# Static FP8 with calibrated activation scales. Needs a small calibration set
# that resembles production traffic.
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LEN = 2048

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)

ds = ds.map(preprocess, remove_columns=ds.column_names)

recipe = QuantizationModifier(targets="Linear", scheme="FP8", ignore=["lm_head"])
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
model.save_pretrained("Llama-3.1-70B-FP8-static", save_compressed=True)
```

Serving either offline checkpoint needs no `--quantization` flag — vLLM detects the `compressed-tensors` config and dispatches the FP8 kernels automatically:

```bash
vllm serve ./Llama-3.1-70B-FP8-static \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 8 \
  --max-model-len 8192
```

For FP4 on Blackwell, the path today runs through NVIDIA's TensorRT Model Optimizer (`modelopt`), which produces NVFP4 checkpoints. The API mirrors the pattern — pick a config, run a calibration forward loop, export:

```python
# NVFP4 weight+activation quantization with TensorRT Model Optimizer.
# Requires a Blackwell GPU to actually run the FP4 kernels at serving time.
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

# NVFP4: 4-bit E2M1 elements, per-16 FP8 block scale, per-tensor FP32 scale.
config = mtq.NVFP4_DEFAULT_CFG

def calib_loop(model):
    for batch in calib_dataloader:  # a few hundred representative sequences
        model(**batch)

model = mtq.quantize(model, config, forward_loop=calib_loop)
export_hf_checkpoint(model, export_dir="Llama-3.1-70B-NVFP4")
```

Older tooling you will still see referenced: **AutoFP8** was the original standalone FP8 quantizer; it is functional but effectively superseded by `llm-compressor`, which unifies FP8, INT8, and INT4 under one recipe API and emits the `compressed-tensors` format vLLM prefers. Its API is a useful contrast because it makes the two FP8 knobs — weight granularity and activation scaling — explicit:

```python
# Legacy AutoFP8 path. Shown for pipelines that still depend on it; new work
# should use llm-compressor. activation_scheme="static" needs calibration data;
# "dynamic" computes per-token activation scales at runtime and needs none.
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

quant_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static",       # or "dynamic" for calibration-free serving
)
model = AutoFP8ForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct", quant_config,
)
model.quantize(calibration_tokens)    # a few hundred tokenized sequences
model.save_quantized("Llama-3.1-70B-FP8")
```

If you are starting fresh, use `llm-compressor` for FP8/INT and `modelopt` for FP4; reach for AutoFP8 only to reproduce an existing pipeline.

A word on mixture-of-experts models, since the largest served models are now MoE. The parameter count is dominated by the expert FFN weights, so that is where the quantization win lives — quantize the experts to FP8 (or FP4) and you capture nearly all of the memory saving. The router/gate network is tiny and disproportionately sensitive, so it belongs on the `ignore` list in high precision; a mis-rounded routing logit sends a token to the wrong expert, which is a far worse error than a slightly noisy activation. The same `ignore=["lm_head", "*gate*"]`-style pattern that protects the output projection protects the router, and it costs almost nothing because the router is a rounding error in the FLOP budget.

One implementation detail that trips people up: FP8 KV cache accuracy improves noticeably if you calibrate the KV scales rather than using a fixed default, because keys and values have their own magnitude distributions. vLLM supports loading per-layer KV scales from a checkpoint produced with KV-cache calibration; if you see a larger-than-expected quality drop from `--kv-cache-dtype fp8`, calibrated KV scales are the first thing to try before you conclude FP8 KV is too lossy for your model.

To show the scaling mechanics rather than just call a library, here is the per-token-versus-per-tensor difference in plain PyTorch. This is exactly what the runtime does for activation quantization, and running it on a tensor with a planted outlier makes the accuracy gap visible:

```python
import torch

def quantize_per_tensor(x, n_levels=127):
    # One scale for the whole tensor: hostage to the global max.
    s = x.abs().max() / n_levels
    xq = torch.clamp(torch.round(x / s), -n_levels, n_levels)
    return xq * s, s

def quantize_per_token(x, n_levels=127):
    # One scale per row (token): each token gets its own max.
    s = x.abs().amax(dim=-1, keepdim=True) / n_levels
    xq = torch.clamp(torch.round(x / s), -n_levels, n_levels)
    return xq * s, s

torch.manual_seed(0)
x = torch.randn(4, 4096) * 2.0           # normal activations near |2|
x[1, 137] = 340.0                        # one outlier in row 1, channel 137

xt, _ = quantize_per_tensor(x)
xk, _ = quantize_per_token(x)

def rel_err(a, b, mask):
    return ((a[mask] - b[mask]).abs() / b[mask].abs().clamp_min(1e-6)).mean()

normal = torch.ones_like(x, dtype=torch.bool)
normal[1, 137] = False  # measure error on the non-outlier values only

print(f"per-tensor mean rel err (normal values): {rel_err(xt, x, normal):.4f}")
print(f"per-token  mean rel err (normal values): {rel_err(xk, x, normal):.4f}")
```

Running it prints something close to this — the per-token scale confines the outlier's damage to its own row, so the other three rows keep near-full INT8 precision, while the per-tensor scale lets one value in one row degrade everything:

```console
per-tensor mean rel err (normal values): 0.0170
per-token  mean rel err (normal values): 0.0043
```

The per-token error is roughly 4x lower on the normal values, and that gap widens as the outlier grows or the tensor shares more elements per scale. Swap `amax(dim=-1)` for a block reduction over groups of 32 columns and you have the microscaling mechanism from the FP4 section.

## Measuring the damage: perplexity, KL, and task accuracy

You never ship a quantized model on faith. You measure the drift against the FP16 reference on three complementary metrics, because each catches a different failure. **Perplexity** on held-out text is the cheapest and most sensitive early-warning signal: it moves before task accuracy does, so a large perplexity jump is a red flag even if MMLU looks fine. **KL divergence** (or its cousins — logprob drift, top-token agreement) between the quantized and reference next-token distributions is the most direct measure of "did the model's behavior change," because it compares the actual output distributions token by token rather than a downstream score. **Task accuracy** — MMLU, GSM8K, HumanEval, whatever your users actually rely on — is the metric that ultimately decides ship-or-not, but it is noisy and coarse, so you use it to confirm rather than to detect.

![Before-and-after comparison: FP8 E4M3 keeps perplexity, MMLU, and KL drift inside the noise floor, while FP4 opens a measurable gap that NVFP4's finer scaling narrows but does not close](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-8.webp)

The figure states the punchline, and it is the honest summary of where the two formats land. FP8 E4M3 moves all three metrics inside measurement noise: perplexity drifts by a few hundredths, MMLU by well under a point, KL divergence on the order of a hundredth of a nat. For most models and most tasks, FP8 W8A8 with FP8 KV is indistinguishable from the baseline in any evaluation your users would notice. FP4 is a different regime: MXFP4 post-training quantization typically costs a few tenths of a point of perplexity and one to three points of task accuracy, which is real and sometimes unacceptable. NVFP4's finer per-16 FP8 block scale recovers roughly half of that gap in post-training quantization, and quantization-aware training or distillation can close most of the rest — but "close most of the rest" still means FP4 asks for an accuracy budget and a measurement discipline that FP8 does not. Frame every one of these numbers as representative of published reports and your own model's mileage will vary; the *shape* — FP8 lossless, FP4 costly-but-recoverable — is the durable claim, not any single decimal.

Here is a harness you can run today to get your own numbers. It computes per-token KL divergence and top-1 agreement between two vLLM engines — the FP16 reference and a quantized variant — on a set of prompts, which is a far tighter signal than a benchmark score:

```python
# Compare a quantized model's next-token distribution against the FP16 reference.
# KL divergence and top-1 agreement catch behavioral drift that task scores miss.
import torch
from vllm import LLM, SamplingParams

PROMPTS = [ ... ]  # a few hundred representative production-style prompts

def logprobs_for(model_path, prompts, dtype=None, kv_dtype="auto"):
    kwargs = dict(tensor_parallel_size=8, max_model_len=4096, kv_cache_dtype=kv_dtype)
    if dtype:
        kwargs["quantization"] = dtype
    llm = LLM(model=model_path, **kwargs)
    # logprobs=20 returns the top-20 token logprobs at each generated position.
    sp = SamplingParams(max_tokens=64, temperature=0.0, logprobs=20)
    return llm.generate(prompts, sp)

ref = logprobs_for("meta-llama/Llama-3.1-70B-Instruct", PROMPTS)          # FP16
q8  = logprobs_for("meta-llama/Llama-3.1-70B-Instruct", PROMPTS,
                   dtype="fp8", kv_dtype="fp8_e4m3")                       # FP8

def kl_and_agreement(ref_out, q_out):
    kls, agree, n = 0.0, 0, 0
    for r, q in zip(ref_out, q_out):
        for rt, qt in zip(r.outputs[0].logprobs, q.outputs[0].logprobs):
            # Align on the reference's top-20 token set.
            ids = list(rt.keys())
            rp = torch.tensor([rt[i].logprob for i in ids])
            qp = torch.tensor([qt[i].logprob if i in qt else -30.0 for i in ids])
            rp, qp = rp.log_softmax(0), qp.log_softmax(0)
            kls += torch.sum(rp.exp() * (rp - qp)).item()   # KL(ref || quant)
            agree += int(max(rt, key=lambda i: rt[i].logprob)
                         == max(qt, key=lambda i: qt[i].logprob))
            n += 1
    return kls / n, agree / n

kl, top1 = kl_and_agreement(ref, q8)
print(f"FP8  mean KL(ref||quant): {kl:.4f} nats,  top-1 agreement: {top1:.3%}")

# Extend the same harness to FP4 for a three-way FP16 / FP8 / FP4 comparison.
# Point it at an NVFP4 checkpoint you exported earlier (Blackwell to run FP4 kernels).
q4 = logprobs_for("./Llama-3.1-70B-NVFP4", PROMPTS, kv_dtype="fp8_e4m3")
kl4, top1_4 = kl_and_agreement(ref, q4)
print(f"FP4  mean KL(ref||quant): {kl4:.4f} nats,  top-1 agreement: {top1_4:.3%}")
```

A healthy FP8 result looks like a mean KL in the low hundredths of a nat and top-1 agreement in the high nineties of a percent. When you run the same harness against an FP4 checkpoint, you will see the KL climb and the top-1 agreement drop by a few percent — the quantitative version of the figure's claim. Set a KL threshold and a minimum top-1 agreement as your ship gate and you have turned "does it still work" into a number you can put in CI. The [evaluating-serving-quality-under-load post](/blog/machine-learning/model-serving/evaluating-serving-quality-under-load) extends this into continuous production monitoring, so drift that only shows up on real traffic gets caught after deploy, not just before.

The KL harness is the sharp instrument; perplexity is the cheap early-warning gauge, and it is worth having both. Perplexity is the exponential of the mean negative log-likelihood the model assigns to held-out text — lower means the model is less surprised by real text — and because it integrates over the whole distribution rather than just the top token, it moves before top-1 agreement does. A ten-line loop over the same reference and quantized checkpoints gives you the number:

```python
# Perplexity on held-out text, computed identically for each precision so the
# deltas are comparable. A jump here is an early warning even if top-1 looks fine.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def perplexity(model_path, text, quant=None, stride=512, ctx=2048):
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = {"torch_dtype": "auto", "device_map": "auto"}
    if quant:  # e.g. "fp8" — vLLM-style flags differ; HF loads compressed-tensors directly
        kw["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw).eval()
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    nll, n = 0.0, 0
    for i in range(0, ids.size(1) - 1, stride):
        window = ids[:, i : i + ctx]
        with torch.no_grad():
            out = model(window, labels=window)
        nll += out.loss.item() * (window.size(1) - 1)
        n += window.size(1) - 1
    return float(torch.exp(torch.tensor(nll / n)))

held_out = open("wikitext_sample.txt").read()
print(f"FP16 ppl: {perplexity('meta-llama/Llama-3.1-70B-Instruct', held_out):.3f}")
```

Run it once per precision and record the deltas next to the KL numbers. The pattern you are looking for is the one this post keeps asserting: FP8 moves perplexity by a few hundredths and KL by a few hundredths of a nat, both inside the run-to-run noise; FP4 moves both by an amount you can see, which NVFP4 shrinks but does not erase. If perplexity jumps but KL and top-1 look fine, you have a tail problem — the model is fine on the common tokens and wrong on the rare ones — which is exactly the signature of a few outlier-heavy layers that belong on the `ignore` list.

Pulling the measurements together into the table you actually plan capacity from — precision by scaling by memory by speedup by accuracy, for a 70B-class model, framed as representative:

| Recipe | Scaling | Memory (70B) | Decode speedup | Prefill speedup | Accuracy drift | Hardware |
|---|---|---|---|---|---|---|
| BF16 (baseline) | — | 140 GB | 1.0x | 1.0x | 0 (reference) | any |
| W4A16 INT4 (AWQ) | per-group 128 | 37 GB | ~2–3x | ~1.0x | +0.2–0.4 PPL | any CUDA |
| W8A8 FP8 E4M3 | per-chan / per-tok | 70 GB | ~1.6–2x | ~1.5–1.8x | +0.02–0.05 PPL | Hopper, Ada |
| W8A8 FP8 + FP8 KV | + per-layer KV | 70 GB + ½ KV | ~1.6–2x, 2x ctx | ~1.5–1.8x | +0.03–0.06 PPL | Hopper, Ada |
| W4A4 MXFP4 | per-block 32 | 37 GB | ~2–3x | ~2x | +0.3–0.8 PPL | Blackwell |
| W4A4 NVFP4 | per-16 + FP8 scale | 39 GB | ~2–3x | ~2x | +0.15–0.4 PPL | Blackwell |

Read the table as a frontier, not a ranking. On Hopper, FP8 gives you both memory and compute wins at an accuracy cost that hides in the noise — it is the obvious default. On Ampere, W4A16 gives the memory win only, but that is still the highest-leverage move available since Ampere has no FP8 path. On Blackwell, FP4 doubles the FP8 win again, and NVFP4's finer scaling buys back enough accuracy to make it viable for many workloads — but only if you have measured the drift and decided you can afford it.

#### Worked example: the FP4-versus-FP8 decision on Blackwell

You have a B200 fleet serving a 70B chat model at 12,000 tokens per second on FP8, and product wants to either cut cost per token or raise the ceiling for a traffic surge. FP4 (NVFP4) is on the table because the hardware supports it. Work the decision with a tolerance, not a vibe.

Set an accuracy budget first: say the product owner accepts up to a 1-point MMLU drop and a 2% relative drop in top-1 agreement versus FP16, because the chat surface is forgiving and the win is large. Run the harness. Suppose NVFP4 comes back at a 0.7-point MMLU drop and 1.6% top-1 disagreement — inside budget. Now the throughput math: FP4 roughly doubles the tensor-core rate versus FP8, and it halves weight bytes again (70 GB to ~39 GB), so on a prefill-heavy or large-batch workload you might see aggregate throughput move from 12,000 toward 18,000–22,000 tokens per second, and the freed memory lets you hold more concurrent sequences. If instead NVFP4 had come back at a 2.5-point MMLU drop, it is out — you would stay on FP8 and get your surge headroom from more GPUs or from FP8 KV and a bigger batch, not from spending accuracy you do not have. The discipline is the point: FP4 is a lever you pull *after* the measurement clears a pre-agreed budget, never before.

## Case studies

**DeepSeek-V3 trained natively in FP8.** The most consequential FP8 result is not a serving benchmark but a training one: DeepSeek-V3 (a 671B-parameter mixture-of-experts model) was trained with an FP8 mixed-precision framework, keeping the bulk of the GEMMs in FP8 E4M3 with fine-grained (per-block and per-tile) scaling and selectively higher precision for the accumulation and the sensitive operations. The reason it matters for serving is that a model *trained* in FP8 is trivially *served* in FP8 — there is no post-training quantization gap to worry about, the weights already live on the FP8 grid. It is also the strongest existence proof that E4M3's three mantissa bits, combined with fine-grained scaling, are enough to represent a frontier model's numerics faithfully. The fine-grained scaling detail is the load-bearing one: DeepSeek's framework does not use a single per-tensor scale, it scales in small blocks, which is the same principle this post derives for FP4.

The specifics are worth carrying because they generalize. DeepSeek's reported recipe scales activations per ${1 \times 128}$ tile (a group of 128 channels within a token) and weights per ${128 \times 128}$ block — granularities chosen to keep each scale's region small enough that the FP8 dynamic range covers it without clipping the local outliers. The second detail matters more than the first: because Hopper's FP8 tensor cores accumulate the running dot-product in a register precision narrower than full FP32, the framework periodically promotes the partial sums into FP32 accumulators (a "promotion" every fixed number of MMA steps) to stop accumulation error from compounding across the long inner dimension. That is the same principle the quantized-linear-layer dataflow figure earlier in the post shows as "accumulate in FP32" — DeepSeek is engineering around the reality that the inexpensive-but-narrow accumulate is where FP8 training would otherwise quietly lose its bits. The takeaway for a serving engineer is that the format is only half the story; where and how you accumulate is the other half, and the good news is that inference frameworks already accumulate FP8 GEMMs in FP32 for you.

**Blackwell and NVFP4 accuracy claims.** NVIDIA introduced NVFP4 alongside Blackwell as the format meant to make 4-bit inference production-viable, and the published positioning is that NVFP4 — with its per-16-element FP8 block scale and per-tensor FP32 scale — recovers most of the accuracy that plain MXFP4 loses, landing within roughly a percent of the FP8/BF16 baseline on a range of benchmarks when combined with careful post-training quantization or light quantization-aware training. Treat the exact percentages as vendor-reported and workload-dependent; the durable engineering takeaway is architectural, that the two-level scale (finer block, FP8-precision scale) is what closes the gap, and that FP4's viability rests on the scaling scheme far more than on the 4-bit elements themselves. The hardware doing the microscale in the tensor core is what turns "small" into "fast."

**FP8 as the default in production serving stacks.** Across the open serving ecosystem, FP8 has quietly become the recommended default for Hopper-class deployments. vLLM ships first-class FP8 support for both weights and KV cache and documents near-lossless results on the common instruct models; TensorRT-LLM treats FP8 as a primary path on Hopper; and the `compressed-tensors` / `llm-compressor` toolchain exists specifically to make FP8 checkpoints a routine artifact rather than a research project. The consistent report from these stacks is the one this post opened with: FP8 delivers roughly a 1.5–2x throughput improvement and a halving of weight memory at an accuracy cost that repeatedly measures inside eval noise on well-behaved models. The dissenting data points — models that lose more than expected — almost always trace back to a small number of genuinely outlier-heavy layers, which is exactly the failure mode finer scaling (or leaving those specific layers in higher precision via an `ignore` list) is designed to fix.

**Native low-precision checkpoints are becoming the release format.** The trajectory the DeepSeek result points at is now visible in how models ship: rather than releasing a BF16 checkpoint and leaving each deployer to quantize it, model authors increasingly release the low-precision checkpoint *as the primary artifact* — FP8 weights for Hopper-class serving, and, on Blackwell-era releases, MXFP4 or NVFP4 expert weights for the largest MoE models. The engineering logic is the same one that makes DeepSeek's FP8 free to serve: a checkpoint that was quantized (or trained) with the author's own calibration and their knowledge of which layers are sensitive is strictly better than a downstream post-training pass with a generic calibration set, because the author knows where the outliers are. Treat the specific formats and per-model accuracy numbers here as a moving target — the durable point is that "quantize it yourself" is shifting toward "it already shipped quantized," and the skills in this post move from *producing* the checkpoint to *validating* the one you were handed. You still run the KL and perplexity harness; you are just checking someone else's work instead of your own.

## When to use this (and when not to)

The whole post collapses into a short decision procedure, and it helps to have the shape of it in front of you before the individual recommendations.

![Decision tree for picking a serving precision: GPU generation is the first branch — Blackwell reaches NVFP4 W4A4 or falls back to FP8 depending on the accuracy budget, Hopper runs FP8 W8A8 with FP8 KV, and Ampere or older falls back to INT W4A16 via AWQ or GPTQ](/imgs/blogs/fp8-fp4-low-precision-serving-deep-dive-6.webp)

Read the tree top-down: your silicon narrows the field before accuracy ever enters the picture, and only on Blackwell does the accuracy budget get to decide the leaf. That structure is why "which format" is rarely an open-ended question — the hardware answers most of it for you, and the measurement answers the rest. The three recommendations below are just the branches of that tree spelled out.

**Use FP8 almost always for LLM serving on Hopper or Ada.** If your hardware has FP8 tensor cores, W8A8 FP8 with FP8 KV is close to a free lunch: you halve weight memory, roughly double KV capacity, get both a decode-memory and a prefill-compute win, and pay an accuracy cost that hides in the noise on the vast majority of models. The correct default is to turn it on, run the KL/top-1 harness once to confirm the drift is inside your gate, and ship. The only reasons not to are a model you have specifically measured to be FP8-sensitive (rare, and usually fixable by keeping a few layers high-precision), a regulatory or contractual requirement to serve bit-identical outputs, or a latency-critical single-stream workload so small that the batch never gets big enough to be compute-bound — though even then the memory win alone usually justifies it.

**Use FP4 only with an accuracy budget and Blackwell silicon.** FP4 is not a default; it is a deliberate spend. Reach for it when you are on Blackwell, when throughput or cost-per-token is the binding constraint, and when you have run the measurement and confirmed the accuracy drift is inside a pre-agreed tolerance for *your* task. Prefer NVFP4 over MXFP4 for the finer scaling, and be ready to invest in quantization-aware training if post-training quantization does not clear your budget. Do not use FP4 on pre-Blackwell hardware expecting a speedup — you will get the memory win and an upcast GEMM, which is rarely worth the accuracy risk. And do not use FP4 on accuracy-critical surfaces (medical, legal, code that runs unreviewed) without a much tighter budget and much more measurement than a chat surface needs.

**Do not confuse the memory win with the compute win.** W4A16 (weight-only) buys memory and decode-bandwidth relief but no tensor-core speedup, because the activations stay in FP16 and the GEMM runs in FP16. W8A8 and W4A4 buy both because both operands are low precision. If your bottleneck is prefill or large-batch decode (compute-bound), you need the activations quantized; if it is small-batch decode (memory-bound), weight-only is enough. Match the recipe to the bottleneck, which means profiling first — the [roofline post](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference) tells you which regime you are in.

## Key takeaways

- **Format = a bet on range versus precision.** More exponent bits buy range, more mantissa bits buy precision. E4M3 (3 mantissa bits, max ±448) is the inference default; E5M2 (2 mantissa bits, max ±57344) trades precision for range and mostly belongs in training.
- **FP8 has two payoffs, FP4 doubles both.** Halving bytes per parameter halves the memory-bound decode floor; low-precision *both operands* unlocks a tensor-core GEMM that is ~2x faster per precision halving. Weight-only quantization gets only the first.
- **Scaling granularity is the accuracy knob.** One scale per tensor is cheap but hostage to outliers; per-token and per-block scaling confine an outlier's damage to its own region. Finer scaling drives the *relative* error toward the format's intrinsic floor regardless of magnitude.
- **FP4 only works because of the block scale.** E2M1 is sixteen values; MXFP4 (per-32 power-of-two scale) and NVFP4 (per-16 FP8 scale plus a per-tensor FP32 scale) carry the magnitude so the 4-bit element only stores shape.
- **Quantize three things separately.** Weights (easy, offline, per-channel), activations (harder, dynamic per-token, the price of the compute win), and the KV cache (FP8 KV is often the cheapest win — halves attention memory for near-zero accuracy cost).
- **Calibration moves difficulty around.** SmoothQuant migrates outlier magnitude from activations into weights; AWQ protects salient weight channels. Both let a coarse scale do a fine scale's job — critical for 4-bit, optional for FP8.
- **Match the format to the silicon.** FP8 needs Hopper or Ada; FP4 needs Blackwell. Buying the format without the tensor cores buys you the memory win and an upcast GEMM — half the benefit.
- **Measure drift with KL and top-1 agreement, not just a benchmark.** FP8 lands inside the noise floor; FP4 opens a measurable gap that NVFP4 narrows. Set a KL/agreement threshold as a CI ship gate.

## Further reading

- **FP8 Formats for Deep Learning** (Micikevicius et al., 2022) — the paper that standardized E4M3 and E5M2 and explains the range/precision split and why inference wants E4M3.
- **OCP Microscaling (MX) Formats Specification** — the open standard behind MXFP4/MXFP8: shared block scales, E8M0 scale format, and the block-size choices.
- **SmoothQuant** (Xiao et al., 2022) and **AWQ** (Lin et al., 2023) — the two calibration methods that make W8A8 and W4 weight-only accurate by redistributing quantization difficulty.
- **DeepSeek-V3 Technical Report** — a frontier MoE trained natively in FP8 with fine-grained block scaling; the strongest existence proof that FP8 numerics are enough.
- **vLLM quantization docs** and the **llm-compressor** / **compressed-tensors** repositories — the production toolchain for producing and serving FP8/INT checkpoints, plus **NVIDIA TensorRT Model Optimizer** for NVFP4 on Blackwell.
- Within this series: [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) (the layer above this one), [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization), [GPU-architecture-specific tuning](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving), [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference), [evaluating serving quality under load](/blog/machine-learning/model-serving/evaluating-serving-quality-under-load), and [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different).
