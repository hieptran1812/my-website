---
title: "FP8 and FP4 inference: what the hardware actually gives you"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Unlike int4, FP8 and FP4 are compute wins, not just bandwidth wins — the tensor cores run them natively. Derive the formats from the bits up, see where the advertised 2x and 4x materialize and where they evaporate, and build a scaled fp8 GEMM into nanoserve."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "quantization",
    "fp8",
    "fp4",
    "gpu",
    "pytorch",
    "cuda",
    "ml-systems",
    "throughput",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 42
---

A colleague hands you two checkpoints of Llama-3.1-8B and a claim. The first is int4, and the slide says "4× faster decode." The second is fp8, and the slide says "2× faster." You are serving a chat endpoint at batch 1 — one user, short prompts, long replies — so you load the int4 one and you do get most of that 4×. Great. Then traffic grows, you turn continuous batching up to 64 concurrent requests to keep the GPU busy, and the int4 model's advantage quietly collapses to nothing. Someone suggests the fp8 checkpoint instead. You swap it in and now the *opposite* happens: at batch 1 the fp8 model is barely faster than bf16, but as the batch climbs it holds a clean ~2× the whole way up. Two "faster" models, and each one is fast in exactly the regime where the other is slow.

That is not a benchmark artifact. It is the single most important fact about low-precision inference, and almost every "quantization makes it faster" conversation gets it wrong by flattening it. Weight-only int4, which we built a kernel for in [the dequant-fused GEMM post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly), is a **bandwidth** win: you move fewer weight bytes per token, and at batch 1 decode is bandwidth-bound, so you go faster. But the actual multiply still runs in fp16 on the tensor cores — you did not make the GPU compute less. FP8 and FP4 are a different animal. The tensor cores execute them *natively*, at 2× and 4× the FLOP rate of fp16. They are **compute** wins. And compute is exactly what you become bound on once the batch is large enough — precisely where int4's bandwidth win has already evaporated.

![Side-by-side flow contrasting weight-only int4 as a bandwidth-only win that dies at large batch against native fp8 and fp4 which also lift the compute ceiling](/imgs/blogs/fp8-and-fp4-inference-what-the-hardware-actually-gives-you-1.webp)

This post is about what the silicon actually delivers. By the end you will be able to read an E4M3 number down to its bits, derive why E4M3 is for weights and E5M2 is for range, explain why 4-bit *float* is unusable without block scaling and exactly what NVFP4 does to fix it, and — the load-bearing skill — predict from a roofline whether a given format's advertised speedup will show up on *your* batch size and *your* GPU or quietly not. You will also write `nanoserve/kernels/fp8_gemm.py`: per-token and per-channel fp8 quantization, a scaled GEMM that feeds the fp8 tensor-core path, and a harness that measures quantization error so you can see per-tensor scaling fall apart on an outlier tensor with your own eyes. This is the compute-precision-and-hardware post; quantizing the *cache* rather than the weights is its own thing, and it lives in [the KV-cache quantization post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff).

One promise, the same one from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a vendor spec or an official post with a link, or framed as something you reproduce yourself with a named script and an expected range. The results tables carry a `Source` column. Format math — bits, ranges, byte ratios — is the honest part, because it is arithmetic. End-to-end throughput is the hard part, and those numbers stay cited.

---

## 1. Two kinds of win, and why the difference is the whole story

Start with the two rooflines, because they *are* the argument. [The roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) says a kernel is limited by whichever is slower: reading its bytes from HBM, or doing its FLOPs on the arithmetic units. A matmul has an **arithmetic intensity** — FLOPs performed per byte moved — and the hardware has a **ridge point**, the intensity at which the two limits are equal. Below the ridge you are memory-bound and the compute units idle waiting for bytes; above it you are compute-bound and the memory system idles waiting for the math to finish.

A batch-1 decode step is deep in memory-bound territory. It streams the whole weight matrix through the units to produce one token, reusing each weight exactly once, so its arithmetic intensity is about 1 FLOP per byte — as memory-bound as a kernel gets. The step time is therefore set by bytes over bandwidth:

$$
t_{\text{decode}} \approx \frac{\text{weight bytes}}{\text{HBM bandwidth}}
$$

Now the two formats do different things to that equation. Weight-only int4 shrinks the **numerator**: a 4-bit weight is a quarter the bytes of a bf16 weight, so you read ~4× less and, on the memory-bound path, go ~4× faster. But it does nothing to the *compute* — the packed int4 weights are unpacked to fp16 in registers and the multiply happens in fp16, at the fp16 tensor-core rate. The tensor cores never see int4 as an operand type for this path. FP8 shrinks the numerator too, but only by 2× (8 bits vs 16). Its real trick is elsewhere: the fp8 operands go **straight into the tensor cores**, which have a dedicated fp8 datapath that runs at twice the fp16 FLOP rate. FP8 lowers the memory term *and* raises the compute ceiling. FP4 on Blackwell does the same one rung further down — 4× fewer bytes and 4× the fp16 FLOP rate.

This is why the figure above splits into two columns that look similar on the left (both "read less") and diverge on the right. The int4 column's compute node is greyed out — the cores run at 1× — and its win is labeled "gone at big batch." The fp8/fp4 column's compute node is a win, and the advantage carries into the large-batch regime. Hold onto the one-sentence version: **int4 makes you read less; fp8 and fp4 make you read less *and* compute faster.** Everything downstream is a consequence of that.

The reason the distinction *matters operationally* — rather than being a piece of trivia — is that a real server does not run at batch 1. It runs continuous batching (the loop we wrote in [the continuous-batching post](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)) to keep the expensive GPU busy, which means it lives at whatever batch size the traffic and your `max-num-seqs` allow. The higher that batch, the more the roofline slides toward compute-bound, and the more the choice between "bandwidth win" and "compute win" decides your throughput. So before we touch a single bit of E4M3, we have to be able to say where the batch lands on the roofline. That is the next section.

---

## 2. The critical batch: where int4 dies and fp8 lives

Here is the mechanism, derived rather than asserted. Take a linear layer, the workhorse of a transformer: $Y = XW$ with activations $X \in \mathbb{R}^{B \times K}$ (B tokens in the batch, K input features) and weights $W \in \mathbb{R}^{K \times N}$ (N output features). The matmul does $2BKN$ floating-point operations (a multiply and an add per inner-product term). The weights, in bf16, are $2KN$ bytes, and at reasonable batch sizes they dominate the byte traffic — the activations are a thin $2BK$ and $2BN$ on the sides. So the arithmetic intensity, FLOPs per weight byte, is:

$$
\text{AI} \approx \frac{2BKN}{2KN} = B \;\; \text{FLOP/byte}
$$

The intensity of the layer is *just the batch size*. That is a beautifully simple result and it is the hinge of this entire post: as you batch more tokens together, each weight byte you read gets reused across more tokens, and the intensity climbs one-for-one with B. Batch 1 sits at intensity 1 (memory-bound); batch 256 sits at intensity 256.

Now put it against the ridge point. The ridge is peak FLOP rate divided by peak bandwidth. On an A100 80GB, that is about $312 \times 10^{12}$ bf16 FLOP/s over about $2.0 \times 10^{12}$ B/s, which is a ridge near 156 FLOP/byte ([the same A100 figures the HPC series uses](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)). On an H100 SXM it is roughly $989 \times 10^{12}$ over $3.35 \times 10^{12}$, a ridge near 295. Since the layer's intensity equals B, the layer crosses from memory-bound to compute-bound when B crosses the ridge — around batch 150 on an A100, around batch 300 on an H100. Below that batch you are reading bytes; above it you are grinding FLOPs.

![Timeline of the batch axis showing int4 leading while memory-bound near batch 1, both crossing the compute ridge near batch 150, and int4 turning to a net loss while fp8 holds two times past it](/imgs/blogs/fp8-and-fp4-inference-what-the-hardware-actually-gives-you-2.webp)

Trace the two formats along that batch axis, which is what the figure above does. Near batch 1, both are memory-bound; int4 moves 4× fewer bytes and fp8 moves 2× fewer, so on pure byte count *int4 is actually the faster of the two at decode*. As B climbs toward the ridge, the weight read amortizes over more tokens and the memory term shrinks relative to the compute term for everyone. Past the ridge you are compute-bound, and now only the *compute* rate matters. int4's compute rate is the fp16 rate — it never had a native matmul — so past the ridge int4 buys you nothing, and because it still pays a small dequantization cost on the compute path, it goes slightly *negative*: the sibling int4 post derives a ~0.9× (net slowdown) at batch 256. fp8's compute rate is 2× fp16, natively, so past the ridge fp8 keeps a clean ~2×. The curves cross. The format that wins at batch 1 loses at batch 256, and vice versa.

#### Worked example: fp8 vs int4 at batch 1 (the decode floor on a 4090)

Count the bytes for one decode step of Llama-3.1-8B (8.03B parameters) at batch 1, and turn them into a floor on tokens per second using an RTX 4090's 1008 GB/s of memory bandwidth ([NVIDIA's RTX 4090 spec](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)):

- **bf16 weights:** $8.03 \times 10^9 \times 2 = 16.06$ GB. Floor $= 16.06 / 1008 = 15.9$ ms, so about 63 tok/s.
- **fp8 weights:** $8.03 \times 10^9 \times 1 = 8.03$ GB. Floor $= 8.03 / 1008 = 8.0$ ms, so about 126 tok/s — a 2× byte ratio, 2× the ceiling.
- **int4 weights (with group scales):** about 4.14 GB. Floor $= 4.14 / 1008 = 4.1$ ms, so about 243 tok/s — a 3.9× byte ratio.

At batch 1, int4's ceiling (243 tok/s) is nearly double fp8's (126 tok/s), purely because it moves half as many bytes. If your workload is genuinely single-stream decode, int4 is the better bandwidth play and fp8's native compute is dead weight. *(Source: derived; the tok/s are ceilings, and real overheads land an 8B bf16 model in the 40–60 tok/s band on a 4090.)* The point of the example is not the absolute numbers; it is that "fp8 is 2× and int4 is 4×" is a batch-1 statement, and batch 1 is the regime where compute wins are worthless.

#### Worked example: the critical batch where the ranking flips

Take an A100 at batch 256, well past its ridge of ~156, so the linear layers are compute-bound. Now bytes barely matter and the FLOP rate is everything:

- **bf16:** runs at the A100's bf16 tensor rate — call it the 1× baseline.
- **int4:** the matmul still runs in fp16 (there is no native int4 tensor path here), so it runs at the same 1× — and after the per-step dequant overhead, slightly *below* it. The int4 post derives ~0.9× at batch 256. *(Source: derived, cross-linked.)*
- **fp8 on an H100:** runs on the native fp8 tensor path at ~2× the bf16 rate, and stays there as batch grows. *(Source: derived from the 1979 vs 989 dense-TFLOP/s ratio in the [Hopper whitepaper](https://resources.nvidia.com/en-us-tensor-core).)*

So the same two checkpoints that ranked int4 > fp8 at batch 1 rank fp8 > int4 at batch 256. There is no single "faster format." There is a batch size, a ridge point, and which side of it you are on. Every honest fp8-vs-int4 comparison has to name the batch. If it does not, it is measuring one regime and quietly generalizing to both — which is exactly how you end up serving the int4 model into a high-concurrency workload and wondering why the "4× faster" checkpoint is slower than bf16.

One more consequence, because it bites A100 owners specifically: an A100 has **no fp8 tensor path at all** (it is Ampere; fp8 arrived with Hopper). So at batch 256 an A100 user is stuck. int4 has stopped helping, and the "just use fp8" escape hatch does not exist on the silicon. Their only compute-side rescue is a newer GPU. We will make that hardware gate explicit in Section 6; it is the reason the format question and the GPU question cannot be separated.

---

## 3. The formats from the bits up

To reason about where fp8 and fp4 break, you have to be able to read them at the bit level, so let us build them from the bottom. A floating-point number is a sign bit, some **exponent** bits that set its dynamic range, and some **mantissa** bits that set its precision, decoded (for normal numbers) as

$$
\text{value} = (-1)^s \cdot \left(1 + \frac{m}{2^{M}}\right) \cdot 2^{\,e - \text{bias}}
$$

where $M$ is the number of mantissa bits and the exponent is shifted down by a fixed bias so the format can reach both large and small magnitudes. The exponent width buys reach; the mantissa width buys resolution. At 8 bits there is almost nothing to spend, so the two fp8 formats split the leftover four bits differently. This is the same bit-splitting story the training-side post tells in full — [numerical formats and mixed precision](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8) — so here I will keep it tight and inference-focused.

**E4M3** is 1 sign, 4 exponent, 3 mantissa, bias 7. Work its edges out by hand. The largest finite value would be $(1 + 7/8)\cdot 2^{15-7} = 1.875 \cdot 256 = 480$, except that the all-ones exponent with all-ones mantissa is reclaimed as NaN in the OCP `E4M3` (finite) variant, so the true max is the next code down, $(1 + 6/8)\cdot 2^8 = 1.75 \cdot 256 = 448$. The smallest normal is $2^{1-7} = 2^{-6} = 0.015625$, and subnormals reach down to $(1/8)\cdot 2^{-6} = 2^{-9} \approx 0.00195$. The step at 1.0 — the gap between consecutive representable values there — is $2^{-3} = 0.125$, three mantissa bits' worth. Its whole dynamic range, from smallest subnormal to max, is $448 / 0.00195 \approx 2.3 \times 10^{5}$, about 18 binades.

**E5M2** is 1 sign, 5 exponent, 2 mantissa, bias 15 — it trades a mantissa bit for an exponent bit. Max is 57,344, smallest normal is $2^{-14} \approx 6.1 \times 10^{-5}$, and the step at 1.0 is $2^{-2} = 0.25$, coarser than E4M3. So E5M2 reaches much higher and much lower but resolves half as finely between consecutive powers of two.

That single trade decides their jobs. Inference weights and activations are *bounded* quantities — you calibrate them, they sit in a known range — and there you want resolution, so E4M3 with its extra mantissa bit is the default for the forward-pass GEMM. E5M2's extra exponent bit is for values that span a wild range and where losing a mantissa bit is survivable: gradients in fp8 *training*, and the occasional place in inference where range beats precision. For the weight-and-activation matmul this post is about, E4M3 is the format; E5M2 is a supporting actor.

![Matrix comparing E4M3, E5M2, four-bit E2M1, and a block-scaled NVFP4 across their sign-exponent-mantissa split, maximum value, step near one, and role](/imgs/blogs/fp8-and-fp4-inference-what-the-hardware-actually-gives-you-3.webp)

**FP4** — the format NVIDIA's Blackwell tensor cores run natively — is `E2M1`: 1 sign, 2 exponent, 1 mantissa, bias 1. With so few bits you can just enumerate every value it represents. The normals are $(1 + m/2)\cdot 2^{e-1}$ for $e \in \{1,2,3\}$ and $m \in \{0,1\}$, plus one subnormal at $0.5$ and a zero:

$$
\{0,\; 0.5,\; 1.0,\; 1.5,\; 2.0,\; 3.0,\; 4.0,\; 6.0\}
$$

and their negatives. That is the *entire* set of magnitudes a 4-bit float can name: eight of them, topping out at 6.0. There is no code for 5, none for 2.5, none for 10. This is the number set you are rounding every weight in the network onto, which should already worry you — and it is why the next section exists. Before that, the animated figure below shows the rounding happening on the E4M3 grid, where there are many more codes but the same underlying behavior: a real value falls onto the grid and snaps to the nearest representable point, and the grid gets sparser as you go right because a float keeps roughly *constant relative* precision — the absolute gaps grow with magnitude.

<figure class="blog-anim">
<svg viewBox="0 0 720 260" role="img" aria-label="An fp16 value of 2.9 drops onto the E4M3 grid and snaps to the nearest representable point at 3.0, leaving a quantization error of 0.1; the grid ticks are twice as dense below 2 as above it" style="width:100%;height:auto;max-width:860px">
<title>Rounding to E4M3: the value 2.9 snaps to the nearest representable grid point 3.0, and the grid spacing doubles past 2.0 because a float keeps constant relative precision.</title>
<style>
.e43-axis{stroke:var(--text-secondary,#6b7280);stroke-width:2}
.e43-tick{stroke:var(--border,#9ca3af);stroke-width:1.5}
.e43-hit{stroke:var(--accent,#6366f1);stroke-width:3}
.e43-true{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;stroke-dasharray:4 4}
.e43-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.e43-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.e43-acc{font:700 14px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.e43-dot{fill:var(--accent,#6366f1)}
@keyframes e43-snap{0%{transform:translate(0px,-108px);opacity:0}12%{opacity:1}44%{transform:translate(0px,0px);opacity:1}60%{transform:translate(20px,0px);opacity:1}92%{transform:translate(20px,0px);opacity:1}100%{transform:translate(20px,0px);opacity:0}}
.e43-fall{animation:e43-snap 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.e43-fall{animation:none;transform:translate(20px,0px);opacity:1}}
</style>
<text class="e43-lbl" x="360" y="28">E4M3 representable grid — a float keeps constant relative precision</text>
<line class="e43-axis" x1="40" y1="190" x2="690" y2="190"/>
<line class="e43-tick" x1="60"  y1="150" x2="60"  y2="190"/>
<line class="e43-tick" x1="85"  y1="160" x2="85"  y2="190"/>
<line class="e43-tick" x1="110" y1="160" x2="110" y2="190"/>
<line class="e43-tick" x1="135" y1="160" x2="135" y2="190"/>
<line class="e43-tick" x1="160" y1="150" x2="160" y2="190"/>
<line class="e43-tick" x1="185" y1="160" x2="185" y2="190"/>
<line class="e43-tick" x1="210" y1="160" x2="210" y2="190"/>
<line class="e43-tick" x1="235" y1="160" x2="235" y2="190"/>
<line class="e43-tick" x1="260" y1="150" x2="260" y2="190"/>
<line class="e43-tick" x1="310" y1="160" x2="310" y2="190"/>
<line class="e43-tick" x1="360" y1="160" x2="360" y2="190"/>
<line class="e43-tick" x1="410" y1="160" x2="410" y2="190"/>
<line class="e43-hit"  x1="460" y1="146" x2="460" y2="190"/>
<line class="e43-tick" x1="510" y1="160" x2="510" y2="190"/>
<line class="e43-tick" x1="560" y1="160" x2="560" y2="190"/>
<line class="e43-tick" x1="610" y1="160" x2="610" y2="190"/>
<line class="e43-tick" x1="660" y1="150" x2="660" y2="190"/>
<text class="e43-sub" x="60"  y="210">1.0</text>
<text class="e43-sub" x="160" y="210">1.5</text>
<text class="e43-sub" x="260" y="210">2.0</text>
<text class="e43-sub" x="360" y="210">2.5</text>
<text class="e43-acc" x="460" y="210">3.0</text>
<text class="e43-sub" x="560" y="210">3.5</text>
<text class="e43-sub" x="660" y="210">4.0</text>
<text class="e43-sub" x="150" y="238">step 0.125 below 2.0</text>
<text class="e43-sub" x="470" y="238">step 0.25 above 2.0</text>
<line class="e43-true" x1="440" y1="70" x2="440" y2="190"/>
<text class="e43-sub" x="440" y="60">fp16 · 2.9</text>
<circle class="e43-dot e43-fall" cx="440" cy="150" r="9"/>
<text class="e43-acc" x="560" y="128">snap to 3.0 · err 0.10</text>
</svg>
<figcaption>The value 2.9 falls onto the E4M3 grid and snaps to the nearest representable point, 3.0; because a float keeps roughly constant relative precision, the ticks are twice as far apart above 2.0 as below it.</figcaption>
</figure>

Notice what the animation is really showing, because it is the property that separates a *float* format from an *int* one. int4 (which we quantized weights to in the sibling post) has a *uniform* grid — evenly spaced levels within a group, scaled by one number. E4M3's grid is *non-uniform*: fine near zero, coarse near 448, with the step doubling every binade. That means fp8 spends its precision proportionally — a fixed *relative* error everywhere, roughly one part in sixteen — rather than a fixed absolute error. For quantizing tensors whose values span orders of magnitude, that is a genuinely different, and often better, error profile than uniform int8. It is also why the per-tensor-scale failure mode for fp8 is about *dynamic range* (values falling off the bottom of the grid to zero), not about wasted uniform levels. Keep that in mind; Section 5 turns on it.

Here is the full format table, every entry either arithmetic you can redo or a spec you can look up:

| Format | Bits S/E/M | Max finite | Min normal | Step at 1.0 | Dynamic range | Native tensor path | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 1 / 8 / 7 | ~3.4e38 | ~1.2e-38 | ~7.8e-3 | ~76 decades | Ampere+ | derived / cited |
| fp16 | 1 / 5 / 10 | 65,504 | ~6.1e-5 | ~9.8e-4 | ~9 decades | Volta+ | derived / cited |
| E4M3 | 1 / 4 / 3 | 448 | 0.0156 | 0.125 | ~18 binades | Hopper, Ada, Blackwell | derived / cited |
| E5M2 | 1 / 5 / 2 | 57,344 | 6.1e-5 | 0.25 | ~32 binades | Hopper, Ada, Blackwell | derived / cited |
| FP4 E2M1 | 1 / 2 / 1 | 6.0 | 1.0 | 0.5 | 8 magnitudes | Blackwell | derived / cited |

The `Dynamic range` column is the one to stare at. E4M3 spans about 18 binades — plenty for a calibrated tensor. FP4's normals span from 1.0 to 6.0, and with its lone subnormal at 0.5 it names eight magnitudes total. A tensor whose values range over more than a factor of ~12 simply cannot be represented in raw FP4 without either clipping the top or flushing the bottom to zero. That is not a tuning problem; it is a counting problem. Which is the entire reason 4-bit float is useless on its own and needs the trick in the next section.

---

## 4. Block scaling: what makes four-bit float usable

If FP4 can only name eight magnitudes, how does anyone run a model in it? The answer is that you never quantize the whole tensor with one scale. You cut the tensor into small **blocks** — 16 or 32 contiguous values — and give each block its *own* scale factor, stored alongside it. Each FP4 code then only has to encode where its weight sits *inside its block's* narrow range, and a block of 16–32 nearby weights typically spans well under a factor of 12, which eight magnitudes can cover. The shared scale carries the block's absolute magnitude; the 4-bit codes carry only the relative shape. This is **microscaling**, and it is what turns FP4 from a curiosity into something a frontier model actually serves in.

![Graph showing a block of sixteen weights branching to a block absmax and to the individual weights, the absmax setting a shared scale, then the scale and weights merging into E2M1 codes and a dequantized tile](/imgs/blogs/fp8-and-fp4-inference-what-the-hardware-actually-gives-you-4.webp)

The mechanism is exactly what the figure traces. For a block of weights $\{w_i\}$, compute the block's absolute maximum, set a shared scale so that this max maps to the top of the FP4 range ($s = \text{absmax} / 6$), divide every weight by $s$, and snap the result to the nearest E2M1 grid point to get a 4-bit code. To dequantize, multiply the code back by $s$. The reconstructed weight is $\hat{w}_i = s \cdot \text{code}_i$, and its relative error is bounded by the FP4 grid spacing *within the block's range*, not across the whole tensor's range. The block scale did the heavy lifting of spanning magnitudes; the codes only handle the fine structure.

The two microscaling formats that matter differ in exactly how they store that scale, and the difference is a precision-vs-overhead trade you can compute:

- **MXFP4** (the OCP Microscaling standard): block size **32**, scale is an `E8M0` — an 8-bit *pure power-of-two* exponent, no mantissa. Overhead is 8 bits per 32 weights, so the effective cost is $4 + 8/32 = 4.25$ bits per weight. The scale can only be a power of two, which is coarse but cheap.
- **NVFP4** (NVIDIA's Blackwell format): block size **16**, scale is an `E4M3` fp8 value (a real fractional scale, not just a power of two), plus a second per-tensor `FP32` scale on top. Overhead is 8 bits per 16 weights, so the effective cost is $4 + 8/16 = 4.5$ bits per weight.

NVFP4 spends a quarter-bit more per weight than MXFP4 and gets two things for it: a smaller block (16 vs 32, so the shared scale fits its values more tightly) and a higher-resolution scale (E4M3's mantissa lets the scale land between powers of two instead of snapping to them). Both reduce the quantization error, which is why NVIDIA chose the more expensive option for the format its hardware runs natively. The lesson generalizes: **the block scale is where four-bit formats spend their real accuracy budget**, and smaller-block, higher-precision scales cost bits but buy quality. When someone says "we run the model in FP4," the interesting question is never the 4-bit codes — it is the block size and the scale format, because that is what actually determines whether the model survived.

#### Worked example: the memory a 4-bit Llama-3.1-8B really takes

Take the 8.03B-parameter model and cost it in each scheme, so the "4-bit" marketing meets the arithmetic:

- **Raw 4 bits (no scales, unusable):** $8.03 \times 10^9 \times 0.5 = 4.02$ GB.
- **MXFP4 (4.25 bits):** $8.03 \times 10^9 \times 4.25 / 8 = 4.27$ GB.
- **NVFP4 (4.5 bits):** $8.03 \times 10^9 \times 4.5 / 8 = 4.52$ GB.
- **fp8 (8 bits):** $8.03 \times 10^9 \times 1 = 8.03$ GB.
- **bf16 (16 bits):** $16.06$ GB.

So a "4-bit" model is really a 4.25–4.5-bit model once you count the scales, about 4.5 GB rather than 4.0 — the block scales are a ~6–12% tax on the ideal footprint. *(Source: derived.)* That tax is the price of FP4 being representable at all, and it is small next to the 3.5× reduction from bf16. The number to remember is that FP4 roughly halves fp8's footprint and quarters bf16's — and, on Blackwell, does the matmul at 4× the fp16 FLOP rate on top of that.

---

## 5. Where the scale lives: per-tensor, per-channel, per-token

Block scaling for FP4 is one answer to a bigger question that also governs FP8: **at what granularity do you compute the scale?** The choices form a ladder. A **per-tensor** scale is one number for the whole matrix. A **per-channel** scale is one number per row or column. A **per-token** scale is one number per token (per row of the activation matrix). Coarser is cheaper to store and simpler to apply; finer wastes less of the format's range. To see why finer wins, and — more importantly — *which* fine granularity composes with the matmul, we have to derive it.

![Side-by-side contrast of a single per-tensor scale dominated by an outlier against per-token and per-channel scales that keep the outlier local and factor out of the dot product](/imgs/blogs/fp8-and-fp4-inference-what-the-hardware-actually-gives-you-5.webp)

Start with the failure the figure shows on the left. Real LLM activations have **outliers** — a handful of channels whose values are tens of times larger than the rest, a well-documented phenomenon that the calibration post covers in depth and that motivated SmoothQuant and LLM.int8(). Now quantize an activation tensor to E4M3 with one per-tensor scale. The scale is set by the global maximum absolute value, which is the outlier. That is fine for E4M3's *top* — nothing clips — but it drags the *bottom* down: the non-outlier channels, whose values are 30× smaller, now sit 30× lower on the E4M3 grid, and the smallest of them slide beneath E4M3's subnormal floor and flush to zero. One loud channel silences the quiet ones. For FP8 the damage is dynamic-range loss; for FP4, whose entire normal range is a factor of 6, a single per-tensor scale is simply hopeless — hence blocks.

Per-token and per-channel scales fix this by giving each row and column its own window, so an outlier confined to one channel only sets *that* channel's scale and leaves its neighbors alone. But there is a deeper reason those two specific granularities are the right ones, and it is not about outliers — it is about the matmul. Write $Y = XW$ and quantize with a **per-token** scale $s^x_i$ on row $i$ of $X$ and a **per-channel** scale $s^w_j$ on column $j$ of $W$:

$$
X_{ik} \approx s^x_i \, \hat{X}_{ik}, \qquad W_{kj} \approx s^w_j \, \hat{W}_{kj}
$$

where $\hat{X}, \hat{W}$ are the fp8 codes. Substitute into the output:

$$
Y_{ij} = \sum_k X_{ik} W_{kj} \approx \sum_k s^x_i \hat{X}_{ik} \, s^w_j \hat{W}_{kj} = s^x_i \, s^w_j \sum_k \hat{X}_{ik} \hat{W}_{kj}
$$

The two scales are constant across the sum over $k$, so they **pull straight out of it**. The inner sum $\sum_k \hat{X}_{ik}\hat{W}_{kj}$ is a pure fp8 matmul — it runs entirely on the fp8 tensor cores, accumulating in fp32 — and afterward you rescale the fp32 result by the rank-1 outer product $s^x_i s^w_j$, one cheap multiply per output element. This is *exactly* PTPC-FP8's "fused rowwise scaled GEMM," and the derivation shows why the granularity is not arbitrary: the scales must live on the **non-contracted** axes (the token row of $X$, the output-channel column of $W$). Put a scale on the contracted axis $k$ instead — a "per-input-channel" activation scale — and it lands *inside* the sum, where it cannot factor out and cannot be applied after the tensor-core matmul. The matmul's structure dictates the only two fine granularities that are free: per-token and per-channel. That is the whole reason PTPC pairs them.

The vLLM team's [PTPC-FP8 on ROCm post](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) (2025-02-24) built precisely this on AMD MI300X and reports the payoff of doing the rescale *fused* rather than as a separate pass: their fused rowwise-scaled FP8 GEMM runs "up to 2.5× faster than a naive two-step" that dequantizes and then calls a standard GEMM. Same lesson as the int4 kernel — the fusion, not the low precision, is where the speed lives — and the accuracy numbers say the finer granularity nearly closes the gap to bf16 (their table is in the case-studies section). The takeaway for `nanoserve`: if you scale per-tensor you will be simple and occasionally wrong on outlier-heavy layers; if you scale per-token and per-channel you match the matmul's grain and lose almost nothing, at the cost of an outer-product rescale you can fuse into the epilogue.

---

## 6. The hardware support matrix

Everything so far assumed the tensor cores can actually *run* the format. That is a hardware-generation fact, not a software switch, and getting it wrong is how "we'll just enable FP4" turns into a silent 10× slowdown. Here is the matrix, by architecture, cited rather than guessed.

![Tree of GPU architectures branching into an fp8 path for Ada and Hopper, an fp4 path for Blackwell only, and a neither path for Ampere which stays at bf16](/imgs/blogs/fp8-and-fp4-inference-what-the-hardware-actually-gives-you-6.webp)

- **Ampere (A100):** no fp8 tensor path, no fp4. Its tensor cores top out at bf16/fp16 and int8. If you "run fp8" on an A100 you are running an emulation that unpacks to bf16 and computes in bf16 — correct, and no faster than bf16. This is the gate that traps the A100 large-batch user from Section 2.
- **Ada Lovelace (RTX 4090, L4):** native **fp8** (E4M3 and E5M2) tensor cores, at roughly 2× the fp16 rate. **No native fp4.** A 4090 is a perfectly good fp8 inference card and a *non-starter* for native FP4 — run FP4 code on it and it falls back to emulation.
- **Hopper (H100):** native fp8 via the fourth-generation tensor cores and the Transformer Engine. The [Hopper whitepaper](https://resources.nvidia.com/en-us-tensor-core) lists ~1979 dense fp8 TFLOP/s against ~989 bf16 — the clean 2× that Section 2's compute-win argument rests on. No native fp4.
- **Blackwell (B200, GB200):** native **fp4** tensor cores on top of fp8, at roughly 2× the fp8 rate and 4× fp16. Per vLLM's [Blackwell InferenceMAX post](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) (2025-10-09), a B200 is 192 GB of HBM3e at 8 TB/s with native FP4, and FlashInfer supplies the fast fp8/fp4 GEMMs and MoE kernels that feed those cores.

The operational consequence is stark and worth saying plainly: **the FP4 story is Blackwell-only.** Everything you read about 4× FP4 throughput is a B200/GB200 number. On a 4090, an L4, an A100, or an H100, "FP4" is at best a memory-footprint trick (you store 4-bit weights and dequantize to a format the cores *can* run) and gets you the bandwidth win at decode but never the compute win — because there is no 4-bit matmul in the silicon. fp8, by contrast, is broadly available from Ada onward. When you plan a deployment, the format is downstream of the card: pick the GPU, and it tells you which of these wins is even on the table.

The Blackwell numbers are worth citing precisely because they are the clearest public evidence that the compute win is real and carries under load. vLLM's InferenceMAX post frames it as a Pareto improvement (throughput at a given latency, not one peak number) and reports "up to 4× higher throughput at similar latency versus Hopper," with gpt-oss-120B at a 1K/1K chat shape hitting up to 4.3× and Llama-3.3-70B at a 1K/8K reasoning shape up to 3.7× — all attributed to native FP4 plus the FlashInfer kernels and fused AllReduce+RMSNorm+quant. And DeepSeek's own [GB200 write-up on vLLM](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) (2026-02-03) shows the win extending past the matmul into communication: they quantize activations to **NVFP4 before the all-to-all dispatch** in their MoE, cutting the inter-GPU communication volume ~4× versus FP16. That is the same "4-bit is compute *and* bytes" story, applied to the network fabric instead of the tensor cores.

---

## 7. Implementing fp8 in nanoserve

Time to build it. We are adding `nanoserve/kernels/fp8_gemm.py`: fp8 quantization at three granularities, a scaled GEMM that feeds the fp8 tensor-core path where the hardware has one, and an error harness that lets you *see* per-tensor scaling break. PyTorch has had fp8 dtypes (`torch.float8_e4m3fn`, `torch.float8_e5m2`) and a scaled matmul (`torch._scaled_mm`) for a while now; we lean on them so the code is real, not pseudocode. The honesty caveat stands: the quantization and error code runs anywhere (I frame those as reproduce-it-yourself), and any *speed* claim is derived from the FLOP ratio or cited, never a run of mine.

Start with the three quantizers. The pattern is identical — find the absolute max over some axis, set the scale so that max maps to E4M3's 448, divide, and cast — only the axis changes.

```python
# nanoserve/kernels/fp8_gemm.py
import torch

FP8_E4M3_MAX = 448.0  # OCP E4M3 finite max, derived in Section 3

def quantize_per_tensor_fp8(x: torch.Tensor):
    """One scale for the whole tensor. Simplest, and the one that breaks on outliers."""
    amax = x.abs().amax().clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX                       # a single scalar
    x_fp8 = (x / scale).to(torch.float8_e4m3fn)       # cast rounds to the E4M3 grid
    return x_fp8, scale

def quantize_per_token_fp8(x: torch.Tensor):
    """One scale per row (per token) of a [tokens, features] activation matrix."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [tokens, 1]
    scale = amax / FP8_E4M3_MAX
    x_fp8 = (x / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale                                # scale broadcasts over features

def quantize_per_channel_fp8(w: torch.Tensor):
    """One scale per output channel (per column) of a [in, out] weight matrix."""
    amax = w.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)   # [1, out]
    scale = amax / FP8_E4M3_MAX
    w_fp8 = (w / scale).to(torch.float8_e4m3fn)
    return w_fp8, scale
```

Two details that are easy to get wrong. The `clamp(min=1e-12)` guards against a zero-valued row producing a zero scale and a divide-by-zero — real activation rows can be all-zero after a mask. And the axis is the thing: per-token reduces over features (`dim=-1`) because a token is a row; per-channel reduces over the input dimension (`dim=0`) because an output channel is a column. Those are the two non-contracted axes from Section 5's derivation, which is not a coincidence — it is the whole point.

Now the scaled GEMM. `torch._scaled_mm` does the fp8 matmul and applies row/column scales in its epilogue, which is exactly the factored form we derived. It is finicky about layouts (the right operand wants to be column-major, i.e. a transposed contiguous tensor), so wrap it:

```python
def fp8_linear(x: torch.Tensor, w: torch.Tensor,
               per_token: bool = True, out_dtype=torch.bfloat16):
    """
    y = x @ w, computed on the fp8 tensor cores with per-token x-scales and
    per-channel w-scales, rescaled in the epilogue. x is [tokens, in], w is [in, out].
    On Hopper/Ada/Blackwell this runs on the native fp8 datapath; elsewhere PyTorch
    falls back and it is only a memory trick, not a speed one (see Section 6).
    """
    if per_token:
        xq, sx = quantize_per_token_fp8(x)      # sx: [tokens, 1]
    else:
        xq, sx = quantize_per_tensor_fp8(x)     # sx: scalar
    wq, sw = quantize_per_channel_fp8(w)        # sw: [1, out]

    # _scaled_mm wants the second operand column-major: pass w^T as [out, in] and .t() it.
    y = torch._scaled_mm(
        xq, wq,                                 # fp8 operands
        scale_a=sx, scale_b=sw,                 # the s^x_i and s^w_j from the derivation
        out_dtype=out_dtype,                    # accumulate in fp32, return bf16
    )
    return y
```

The scales `scale_a` and `scale_b` are the $s^x_i$ and $s^w_j$ that factored out of the sum in Section 5 — the API is literally implementing $Y = s^x \odot (\hat X \hat W) \odot s^w$, with the fp8 matmul and the fp32 accumulation happening inside the tensor core and the rescale fused into the epilogue. That is the production shape: the low-precision matmul is native, and the scales are a rank-1 correction that costs one multiply per output. (Exact `_scaled_mm` argument shapes and layout constraints shift across PyTorch versions — check the docstring for your build with `help(torch._scaled_mm)`. The *shape* of the call — fp8 in, fp32 accumulate, rescale out — is the stable part.)

Now the harness that earns its place: a per-tensor-vs-per-token error comparison on a tensor with a planted outlier, so you can watch the failure mode from Section 5 instead of taking it on faith. This runs on CPU or GPU, no fp8 hardware needed — it is pure quantize-then-dequantize error.

```python
def dequantize_fp8(x_fp8, scale):
    return x_fp8.to(torch.float32) * scale

def rel_error(a, b):
    return (a - b).norm() / a.norm().clamp(min=1e-12)

def outlier_activation(tokens=64, feats=4096, outlier_chan=37, mag=40.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, feats, generator=g)      # most values ~N(0,1)
    x[:, outlier_chan] *= mag                         # one loud channel, 40x
    return x

if __name__ == "__main__":
    x = outlier_activation()
    xt, st = quantize_per_tensor_fp8(x)
    xk, sk = quantize_per_token_fp8(x)
    print(f"per-tensor rel error : {rel_error(x, dequantize_fp8(xt, st)):.4f}")
    print(f"per-token  rel error : {rel_error(x, dequantize_fp8(xk, sk)):.4f}")
    # per-tensor also flushes small values: count how many non-outlier entries hit zero
    deq_t = dequantize_fp8(xt, st)
    mask = torch.ones_like(x, dtype=torch.bool); mask[:, 37] = False
    zeroed = ((deq_t == 0) & mask & (x.abs() > 0)).float().mean().item()
    print(f"per-tensor small-value flush-to-zero fraction: {zeroed:.3%}")
```

You should expect to see the per-tensor relative error come out several times larger than per-token, and a non-trivial fraction of small non-outlier entries flushed to zero under per-tensor scaling — the quiet channels silenced by the loud one. The exact figures depend on the seed and the outlier magnitude, so *run it and read your own numbers* rather than trusting a printed constant; the qualitative result (per-token is much cleaner on outlier-heavy tensors) is the robust part and it will reproduce across seeds. *(Source: reproduce — run `fp8_gemm.py`.)*

Finally, FP4 — which we can only *emulate*, since I have no Blackwell and neither do most readers. Emulation is still worth writing, because it makes the block-scale mechanism concrete and lets you measure FP4's error even without the hardware. Here is NVFP4-style block quantization onto the E2M1 grid:

```python
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # from Section 3

def quantize_nvfp4_block(w: torch.Tensor, block: int = 16):
    """Emulate NVFP4: per-block E4M3 scale + snap to the E2M1 grid. Returns dequantized
    weights so we can measure error. This is NOT fast anywhere without native fp4 cores —
    it is a correctness/quality tool, and any speed claim would have to be cited."""
    flat = w.reshape(-1, block)                        # [num_blocks, block]
    amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 6.0                                 # 6.0 = E2M1 max
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)  # NVFP4 stores scale in E4M3
    codes = flat / scale
    signs = codes.sign()
    # snap |codes| to nearest E2M1 magnitude
    idx = (codes.abs().unsqueeze(-1) - E2M1_GRID).abs().argmin(dim=-1)
    snapped = signs * E2M1_GRID[idx]
    return (snapped * scale).reshape_as(w)
```

Quantizing the scale itself to E4M3 (`.to(torch.float8_e4m3fn)`) is not a detail to skip — it is the actual NVFP4 storage format, and leaving it in fp32 would flatter your error numbers by pretending the scale is free and infinitely precise. Feed this a weight matrix, compare to the original with `rel_error`, and you can watch NVFP4's error land between fp8's and raw-4-bit's — which is the whole reason the block scale exists. *(Source: reproduce — run the emulator; note that speed on real Blackwell is a separate, cited question this code cannot answer.)*

---

## 8. Where the advertised speedup evaporates

You now have every piece needed to predict, for a given format-GPU-batch triple, whether the number on the slide will show up. The failures are not exotic; they are a handful of regimes, and they are the difference between a deployment that hits the benchmark and one that mysteriously does not.

![Matrix classifying int4 on a 4090, fp8 on H100, fp4 on B200, and fp8 on A100 by what each buys, where it fails, and the verdict](/imgs/blogs/fp8-and-fp4-inference-what-the-hardware-actually-gives-you-7.webp)

**Small batch kills the compute win.** Below the ridge you are bandwidth-bound, and fp8's native 2× FLOP rate is irrelevant because the cores are already idling on memory. At batch 1, fp8 buys you only its 2× byte reduction — less than int4's 4× — so a single-stream chat workload is the one place fp8 looks *weak*. If your whole product is batch-1 latency, the compute-precision formats are solving a problem you do not have; reach for weight-only int4 instead.

**Attention is memory-bound and stays that way.** Even at a large batch, the attention layers read the KV cache, which is a per-token streaming pattern with low arithmetic intensity — they do not become compute-bound the way the big linear layers do. So running the *matmuls* in fp8 speeds up the MLP and projection GEMMs but leaves attention roughly where it was; the KV-cache-side win is a separate lever, quantizing the cache itself, covered in [the KV-cache quantization post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff). Do not expect fp8 weights to fix an attention-bound decode.

**The scaling and dequant overhead is on the compute path.** The rescale, the per-token amax, the cast — they are cheap but not free, and at small batch (where the matmul itself is fast) they are a larger *fraction* of the step. This is the same reason int4 goes slightly negative past the ridge: the quant bookkeeping runs on the very path you were trying to speed up. Fusing it into the GEMM epilogue (as `_scaled_mm` does) is what keeps it from eating the win.

**Quality-driven fallback.** Some layers do not survive fp8 — the calibration post will show which and why — and a robust deployment keeps those in bf16. Every layer you exempt is a layer running at the old rate, so the *realized* speedup is always below the peak format ratio. A model that is "90% fp8, 10% bf16-fallback by FLOPs" cannot hit a full 2×.

**The wrong silicon.** Run FP4 on a 4090 and it emulates through bf16 — correct output, zero compute win, and often *slower* than just running bf16 because of the unpack overhead. Run fp8 on an A100 and the same thing happens. The format has to be in the tensor cores or the compute win is a fiction. This is the single most common way the number on the slide fails to materialize: the slide was a B200 slide and the deployment is an A100.

Here is the decision table, one row per realistic pairing, with the source of each claim:

| Format · GPU | What it buys | Where it fails | Source |
| --- | --- | --- | --- |
| int4 · 4090 | ~3.9× at decode (bandwidth) | net loss past the ridge (~0.9× at B=256) | derived / cited: int4 post |
| fp8 · H100 | ~2× FLOP + ~2× bytes, scales with load | short context / small batch, bf16 can win | derived / cited: Hopper whitepaper |
| fp4 · B200 | ~4× FLOP, up to 4.3× throughput vs Hopper | Blackwell-only; emulated elsewhere | cited: vLLM InferenceMAX |
| fp8 · A100 | nothing native — no fp8 cores | any batch; emulation is bf16-speed | derived: Ampere has no fp8 path |

### Stress-testing the failure modes

Three quick stress tests, each pulling on a thread from above, and each a thing you can actually provoke.

**The outlier that breaks per-tensor FP8.** Run the Section 7 harness with a bigger outlier — bump `mag` from 40 to 200 — and watch the per-tensor relative error climb while per-token barely moves. At `mag=200` the per-tensor scale is set 200× too high for the quiet channels, and E4M3's ~18-binade range means anything more than ~5 orders of magnitude below the outlier is at risk of flushing. This is the exact mechanism the calibration post (the [activation-outliers post](/blog/machine-learning/inference-engineering/activation-outliers-calibration-and-measuring-quality-loss)) attacks with SmoothQuant-style redistribution; here you are seeing why the naive scale fails so you appreciate the fix. *(Source: reproduce.)*

**Running FP4 code on a 4090.** Take the NVFP4 emulator and suppose you wire it into a real forward pass on a 4090. The quantize-dequantize is correct and the *memory* drops, but every matmul still runs in bf16 after dequant — and the per-block unpack adds work the bf16 path never had. So you would measure a model that is smaller in VRAM and *slower* in tokens per second than plain bf16. That is not a bug in your code; it is Section 6's hardware gate. The fix is not software — it is a Blackwell card. *(Source: derived — Ada has no native fp4 path.)*

**Mixing fp8 compute with int4 weights in one model.** These are two different quantization philosophies — fp8 is a *compute* format (operands go into the cores), int4 is a *storage* format (weights get unpacked to fp16). Put an int4-weight layer next to an fp8 layer and you have two kernels with two accumulation paths, two scale conventions, and two different tensor-core setups in the same forward pass. It can be correct — many real deployments are exactly this mixed, int4 for the biggest memory-hog layers at small batch and fp8 for the batched linears — but you own the seam: the int4 layer runs at fp16 compute rate, the fp8 layer at 2×, so your realized throughput is a weighted blend, and the numerics of the two paths must both be validated against the bf16 reference. There is no single "quantized model"; there is a per-layer choice, and the honest throughput number is the blend, not the best layer's ratio. *(Source: derived from the compute-vs-storage distinction.)*

---

## Case studies: real numbers, cited

Three public results, each pinned to its setup so you can weigh it honestly. None of these is my measurement.

**PTPC-FP8 on AMD MI300X.** The vLLM team's [PTPC-FP8 post](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) (2025-02-24) is the cleanest public evidence that per-token-activation, per-channel-weight scaling recovers almost all the accuracy a per-tensor scale loses, while the fused rowwise-scaled GEMM keeps it fast (up to 2.5× vs a naive two-step). Their accuracy table, cited:

| Model | Metric | BF16 | PTPC-FP8 | Source |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | Wikitext perplexity (lower better) | 9.4281 | 9.5093 (+0.86%) | cited: vLLM PTPC-FP8 |
| Llama-3.1-8B | GSM8K accuracy | 73.2% | 70.8% (96.7% of BF16) | cited: vLLM PTPC-FP8 |
| Llama-3.1-70B | GSM8K accuracy | — | 87.3% (slightly above BF16) | cited: vLLM PTPC-FP8 |

The 8B GSM8K drop (73.2% to 70.8%) is the honest cost of fp8 on a smaller model, while the 70B result landing *above* its bf16 comparison is within noise and a reminder that bigger models absorb quantization better. The perplexity gap under 1% is the number that matters for a chat workload. They also report the 70B fused GEMM at 1.01× the throughput of per-tensor fp8 — the finer granularity is essentially free at the kernel level.

**Blackwell FP4 via InferenceMAX.** vLLM's [Blackwell post](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) (2025-10-09) reports up to 4× higher throughput at similar latency versus Hopper across three workload shapes, with gpt-oss-120B at 1K/1K chat reaching up to 4.3× and Llama-3.3-70B at 1K/8K reasoning up to 3.7×, credited to B200's native FP4 tensor cores (192 GB HBM3e at 8 TB/s) plus FlashInfer's fp8/fp4 GEMMs. This is the compute win from Section 1, at production scale, on the only silicon that has the 4-bit datapath. Note the framing: a *Pareto* improvement (a curve that moved at both ends), not a single peak — which is the right way to read any throughput claim.

**DeepSeek on GB200.** DeepSeek's [GB200 write-up on the vLLM blog](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) (2026-02-03) extends FP4 past the matmul: they quantize MoE activations to **NVFP4 before the expert all-to-all**, cutting inter-GPU communication ~4× versus FP16, on a GB200 NVL72 whose 8 TB/s of bandwidth is itself ~1.7× an H200's. It is the same "fewer bits is fewer bytes *and* faster math" idea, applied to the network — and a preview of why FP4 matters most for the largest MoE models, where communication, not compute, is often the wall.

---

## When to reach for this (and when not to)

A decisive read, because "quantize it" is not a plan.

**Reach for fp8** when you run a real server at meaningful concurrency on Ada, Hopper, or Blackwell. Batched serving lives near or past the ridge, where fp8's native 2× is exactly the win you want, and its accuracy cost on an 8B-plus model is typically under a percent with per-token/per-channel scaling. fp8 is the pragmatic default for throughput-oriented LLM serving on modern NVIDIA silicon, and it halves your weight memory as a bonus.

**Reach for FP4** only on Blackwell, and mostly for the largest models — big dense models where the memory saving unlocks a deployment that would not otherwise fit, and large MoEs where FP4 also shrinks the all-to-all. On anything pre-Blackwell, FP4 is a storage format at best; do not expect the compute win. And validate quality harder than for fp8 — 4-bit codes plus a block scale is a real accuracy budget, and it is spent in the scale format, so know whether you are on MXFP4 or NVFP4.

**Prefer weight-only int4** when your workload is genuinely batch-1 latency — single-user chat, an on-device assistant, low-QPS internal tools. There, decode is bandwidth-bound, int4's 4× byte reduction beats fp8's 2×, and the compute win you would pay for in fp8 never activates. The [int4 kernel post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) is the one to read.

**Do not hand-roll any of this for production.** The `nanoserve` kernels here are for understanding — write them, run the error harness, feel the outlier break. But vLLM, SGLang, and TensorRT-LLM ship calibrated fp8/fp4 paths with fused scaled GEMMs, FlashInfer kernels, and per-layer fallback lists that took teams months to get right. Build the toy to know *why* it works and *where* it breaks; serve with [the production engine](/blog/machine-learning/model-serving/quantization-for-llm-serving). The value of writing your own is that you will now read their benchmark tables correctly — you will ask "at what batch, on what GPU, with what scale granularity" instead of believing the headline multiplier.

**And measure honestly.** If you do benchmark a format, do it right: warm up, `torch.cuda.synchronize()` before timing, use CUDA events, hold the clocks steady, and report at a *stated batch size* with an open-loop load — because, as this whole post argued, a tok/s number without its batch size is close to meaningless. A "2× faster" that was measured at batch 1 and deployed at batch 64 is how this all goes wrong. [The reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) is the discipline.

---

## Key takeaways

- **int4 is a bandwidth win; fp8 and fp4 are compute wins.** int4 makes you read fewer weight bytes but still computes in fp16; fp8/fp4 operands run natively on the tensor cores at 2×/4× the fp16 FLOP rate. That single distinction predicts everything else.
- **The batch size decides which win matters.** A linear layer's arithmetic intensity equals its batch size, so it crosses from bandwidth-bound to compute-bound near the GPU's ridge (~150 on A100, ~300 on H100). Below it, int4's 4× beats fp8's 2×; above it, int4 goes to a net loss and fp8 holds 2×.
- **E4M3 keeps mantissa, E5M2 keeps range.** Four exponent bits and three mantissa give E4M3 a max of 448 and fine steps — the weight-and-activation format. Five exponent bits give E5M2 a max of 57,344 for range-hungry values. Both are floats, so their grid is non-uniform: constant relative precision, coarser at large magnitudes.
- **Four-bit float is unusable without block scaling.** FP4 (E2M1) names only eight magnitudes up to 6.0. MXFP4 (block 32, E8M0 scale, 4.25 eff. bits) and NVFP4 (block 16, E4M3 scale, 4.5 eff. bits) restore its dynamic range; the accuracy lives in the block size and the scale format, not the 4-bit codes.
- **Scale granularity is dictated by the matmul.** Per-token activation scales and per-channel weight scales factor out of the dot product as a rank-1 rescale, so they run as a fused fp8 GEMM epilogue. A scale on the contracted axis cannot factor out — which is why PTPC pairs per-token with per-channel.
- **Per-tensor scaling breaks on outliers.** One loud channel sets the scale and flushes the quiet channels toward zero; per-token/per-channel keeps outliers local. This is a dynamic-range failure for fp8 and a fatal one for fp4.
- **FP4 is Blackwell-only.** Ada and Hopper run fp8 natively; only Blackwell (B200/GB200) runs fp4. Ampere (A100) runs neither — fp8/fp4 there is emulation at bf16 speed. The format is downstream of the card.
- **The advertised multiplier is a ceiling, not a promise.** It evaporates at small batch, in attention layers, under quant overhead, with bf16 fallback layers, and completely on the wrong silicon. Always name the batch and the GPU.

---

## Further reading

- [Dequant-fused GEMM: int4 weights on the fly](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) — the bandwidth-win sibling, and where the ~0.9×-at-batch-256 number is derived.
- [GEMM for decode: the skinny-matrix problem](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem) — why batch 1 is a GEMV, the ridge point, and the critical batch in full.
- [KV-cache quantization: fp8, int8, and the accuracy cliff](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) — quantizing the cache instead of the weights; the other half of low-precision serving.
- [Activation outliers, calibration, and measuring quality loss](/blog/machine-learning/inference-engineering/activation-outliers-calibration-and-measuring-quality-loss) — why some channels break low precision, and how to measure the damage honestly.
- [Numerical formats and mixed precision](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8) — the bit-level format story from the training side, in full.
- [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) — the broader landscape of quantization methods and where fp8/fp4 sit in it.
- [PTPC-FP8 on ROCm](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) and [Blackwell InferenceMAX](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) — the vLLM posts behind the cited numbers.
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the series frame and the capstone that benchmarks `nanoserve` end to end.
