---
title: "How Quantization Actually Works: Decoding q4_0, q4_1, and the GGUF Quant Types"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A first-principles, byte-by-byte tour of model quantization: scales and zero-points, why blocks exist, and exactly what q4_0, q4_1, q5_0, Q4_K_M, and the IQ family encode on disk."
tags: ["llm", "quantization", "gguf", "llama-cpp", "q4_0", "q4_k_m", "k-quants", "i-quants", "inference", "model-compression", "imatrix", "bits-per-weight"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 51
---

## When "just use Q4_K_M" stops being enough

Almost everyone who runs a local model has typed `Q4_K_M` into a download URL without knowing what it means. The folklore is good enough most of the time: `Q4_K_M` is "the balanced one," `Q8_0` is "basically lossless," `Q2_K` is "the desperate one." You pick by vibe, the model loads, it talks. Done.

The folklore breaks the moment you step off the happy path. You try to fit a 70B on a 24 GB card and discover `Q2_K` produces fluent nonsense while `IQ2_M` at the *same file size* stays coherent. You quantize your own fine-tune and a perfectly good model develops a stutter on long contexts. You compute an importance matrix on the wrong corpus and a code model forgets how to close a brace. In every one of these cases, the fix comes from knowing what the name actually encodes — and the name encodes a great deal. `q4_0` is not a vibe. It is a 18-byte struct, a reconstruction equation, and a rounding rule, all compressed into four characters.

This post is the decoder ring. We build quantization from the float-to-integer mapping up, and by the end you will be able to look at `Q4_K_M` or `IQ3_XXS` and say, without guessing, how many bytes it spends per block, what arithmetic rebuilds each weight, and which tensors it treats differently. If you want the survey of *methods* — GPTQ, AWQ, SmoothQuant, bitsandbytes — that is a different and complementary post: the [complete practical guide to quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm). This one is about mechanism and on-disk layout.

Here is the gap between the folklore and the machinery:

| What people assume | The folklore view | What is actually true |
|---|---|---|
| `Q4_K_M` is a single "4-bit format" | One uniform 4-bit encoding for the whole model | A *mixing recipe*: most tensors are Q4_K, but the value projection, the down projection, and the output head are bumped to Q6_K |
| The `0` and `1` in `q4_0`/`q4_1` are version numbers | Newer is better, use the higher one | The digit is the *spec*: `_0` stores a scale only (symmetric), `_1` adds a per-block minimum (affine) |
| More bits is always proportionally better | 8-bit is twice as good as 4-bit | Quality climbs steeply from 2 to 4 bits and then nearly flattens; 8-bit buys you almost nothing over 6-bit |
| Quantization rounds the whole tensor with one scale | One scale per weight matrix | One scale per *block* of 32 (legacy) or per sub-block inside a super-block of 256 (K-quants) |
| `IQ` quants are just "smaller K-quants" | Same idea, fewer bits | A different mechanism entirely: a shared non-linear codebook, fit with an importance matrix |
| The model dequantizes to fp16, then does matmul | Weights expand in memory at load time | Weights stay packed; the kernel quantizes the *activation* and does a block-wise integer dot product |

Every row of that table is a section below. Let us start with the one operation underneath all of it.

## The mental model: quantize, store, dequantize

![Pipeline showing a GGUF quant type: fp16 weights split into fixed blocks, a per-block scale and packed integers stored on disk, then dequantized on the fly at matmul time](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. A GGUF quant type is not a clever compression algorithm in the gzip sense. It is a fixed, deterministic, *local* recipe that runs once at quantization time and is undone, piece by piece, every time the model computes:

1. **Partition.** Take a weight tensor stored in fp16 (or bf16) and cut it into fixed-size blocks — 32 weights for the legacy formats, a 256-weight "super-block" for the K-quants.
2. **Fit.** For each block, look at only the 32 (or 256) numbers inside it and compute a small amount of metadata: a scale `d`, and for the affine variants a minimum `m`.
3. **Quantize.** Round each weight to a small integer (4, 5, 6, or 8 bits) relative to that block's scale, and pack those integers tightly.
4. **Store.** Write the block — metadata plus packed integers — to the `.gguf` file. That on-disk byte layout *is* the quant type.
5. **Dequantize.** At inference, the matmul kernel reads a block, reconstructs the weights it needs, and multiplies. Crucially it does not expand the whole tensor back to fp16 in memory; it quantizes the incoming activation to 8-bit and computes a block-wise integer dot product, then accumulates in float. The weights stay small in RAM the entire time — which is the whole point, because LLM inference is memory-bandwidth-bound, not compute-bound.

That last point is worth dwelling on, because it explains *why* weight-only quantization speeds up decoding at all. When you generate one token at a time, the GPU spends most of its cycles waiting for weights to arrive from HBM. A 4-bit weight is a quarter the bytes of an fp16 weight, so it arrives roughly four times faster. The arithmetic to unpack it is nearly free by comparison. This is also why quantization helps decode latency far more than it helps prefill throughput — prefill is compute-bound and already saturates the tensor cores. If you have not internalized that asymmetry, the [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) post is the prerequisite.

Everything else — the difference between `q4_0` and `Q4_K_M`, between K-quants and I-quants — is variation in *steps 2 and 3*: how the per-block metadata is computed and stored, and how the integers are laid out. Master the float-to-integer map and the rest is bookkeeping.

> A quant type is a promise about bytes, not a promise about quality. Quality is what falls out when you keep that promise on a real weight distribution.

## 1. The core operation: mapping floats to integers

**The senior rule of thumb: quantization is just choosing a ruler with sixteen (or 256) evenly spaced tick marks, then snapping every weight to the nearest tick.** The scale `d` is the spacing between ticks; the minimum `m` is where tick zero sits. Pick those two numbers per block and you have defined the entire encoding.

![Number line showing eight weights in one block snapping to the nearest of sixteen integer levels, with the residual gap labeled as quantization error and the quantize/dequantize formulas below](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-2.webp)

Take the figure literally. We have a block of eight weights (real blocks are 32, but eight fits on a page): $\{-0.92, -0.51, -0.18, 0.04, 0.23, 0.40, 0.77, 1.10\}$. We want to store each one in 4 bits, which gives us $2^4 = 16$ integer levels, numbered $0$ through $15$. The recipe with a stored minimum — this is exactly what `q4_1` does — is:

$$
d = \frac{\max - \min}{15}, \qquad m = \min, \qquad q = \mathrm{round}\!\left(\frac{w - m}{d}\right), \qquad \hat{w} = d \cdot q + m
$$

Here $w$ is the original weight, $q \in \{0, \dots, 15\}$ is the 4-bit integer we actually store, and $\hat{w}$ is the reconstructed weight the model sees at inference. For our block, $d = (1.10 - (-0.92))/15 = 0.1347$ and $m = -0.92$. Run every weight through it and the stored integers are $q = [0, 3, 5, 7, 9, 10, 13, 15]$ — exactly the blue levels in the figure. The weight $0.23$ lands at $8.54$ on the ruler and rounds *up* to level $9$; the weight $-0.18$ lands at $5.49$ and rounds *down* to level $5$. The slanted arrows in the figure are those rounding decisions.

The gap between a weight and its tick is the **quantization error**, and it is irreducible. The worst case is when a weight lands exactly halfway between two ticks, so the error is bounded by half a step:

$$
|w - \hat{w}| \leq \frac{d}{2} = 0.067
$$

If you assume weights land uniformly between ticks — a decent approximation inside a small block — the error is uniformly distributed on $[-d/2, d/2]$, which has variance $d^2/12$ and root-mean-square $d/\sqrt{12}$. This single formula drives every design decision later: **error is proportional to the scale $d$, and $d$ is proportional to the block's range.** Shrink the range a block has to cover and you shrink the error. That is the entire argument for blocks, for super-blocks, and for importance matrices.

Here is the whole thing in runnable NumPy, reproducing the figure's numbers exactly:

```python
import numpy as np

def quantize_affine(x, n_bits=4):
    """Asymmetric (affine) quantization, like q4_1:  w ~= d*q + m."""
    qmax = (1 << n_bits) - 1                          # 15 for 4-bit
    lo, hi = float(x.min()), float(x.max())
    d = (hi - lo) / qmax                              # scale == tick spacing
    m = lo                                            # offset == block minimum
    q = np.clip(np.round((x - m) / d), 0, qmax).astype(np.int32)
    return q, d, m, d * q + m                         # q, scale, min, reconstruction

def quantize_symmetric(x, n_bits=4):
    """Symmetric quantization, like q4_0:  w ~= d*(q - 8).  No min stored."""
    half = 1 << (n_bits - 1)                          # 8
    amax = float(np.abs(x).max())
    d = amax / half if amax > 0 else 1.0              # the 8 negative levels set d
    q = np.clip(np.round(x / d) + half, 0, (1 << n_bits) - 1).astype(np.int32)
    return q, d, d * (q - half)

block = np.array([-0.92, -0.51, -0.18, 0.04, 0.23, 0.40, 0.77, 1.10], dtype=np.float32)
q, d, m, xhat = quantize_affine(block)
print("q        =", q)                               # [ 0  3  5  7  9 10 13 15]
print("d, m     =", round(d, 4), round(m, 4))        # 0.1347 -0.92
print("error    =", np.round(xhat - block, 4))
print("max|err| =", round(float(np.abs(xhat - block).max()), 4),
      " d/2 =", round(d / 2, 4))                      # 0.0667  0.0673
```

### Symmetric versus affine, and why the digit matters

Notice the two functions differ in exactly one thing: whether they store a minimum. `quantize_affine` keeps `m` and lets the sixteen levels straddle whatever range the block actually occupies. `quantize_symmetric` throws `m` away and forces the levels to be symmetric around zero — level 8 *is* zero, levels run from $-8d$ to $+7d$. The reconstruction becomes $\hat{w} = d\,(q - 8)$, with no addition.

This is the entire meaning of the trailing digit in the legacy names. `_0` is symmetric: scale only. `_1` is affine: scale plus minimum. Symmetric saves two bytes per block (no `m` to store) and is faster (no add in the inner loop), and it is a great fit for weight distributions that are already centered on zero — which most trained weights are. Affine wins when a block's distribution is lopsided, for instance after a LoRA merge or in the first and last layers, where forcing symmetry wastes half your levels on a range the weights never visit.

### The second-order gotcha: clipping the outlier

There is a subtlety the clean formula hides. `quantize_symmetric` above sets $d = \mathrm{amax}/8$, which means the eight negative levels can reach $-8d = -\mathrm{amax}$ but the seven positive levels only reach $+7d = +\frac{7}{8}\mathrm{amax}$. A positive weight equal to the block's largest magnitude would clip. The real `q4_0` encoder is cleverer: it finds the single weight with the largest *signed* magnitude and sets `d` from that element specifically, so the dominant outlier is represented exactly and everything else rounds around it. The lesson generalizes — every quantizer is a negotiation between representing the outliers and representing the bulk, and the choices a format makes there are exactly where quality is won or lost.

### Rounding is a design choice, not an afterthought

The `round` in $q = \mathrm{round}((w-m)/d)$ hides a decision. Round-half-to-even (banker's rounding, NumPy's default) and round-half-away-from-zero give different integers for any weight that lands exactly on a half-step, and across hundreds of millions of weights those differences accumulate into a measurable bias if you choose wrong. The quantizers that matter go further than nearest-rounding entirely. GPTQ, for instance, does not round each weight independently; it rounds them in sequence and pushes the rounding error of each weight into the not-yet-quantized weights using second-order (Hessian) information, so the *layer's output* error is minimized rather than the per-weight error. GGUF's encoders are simpler — they do an optimized nearest-rounding search per block, sweeping a few candidate scales and keeping the one with the lowest reconstruction error — but the principle holds: the rounding rule is part of the format's quality, and two encoders that emit the same byte layout can write different integers into those bytes. This is why a `Q4_K_M` file from a careful pipeline with an imatrix outscores a lazy one even though both are bit-for-bit the same *format*. The format constrains the bytes; the encoder decides which bytes.

## 2. Why blocks exist: the outlier problem

**The senior rule of thumb: one scale for an entire weight matrix is a disaster, because a single fat outlier sets the scale for ten thousand small weights that then share three of your sixteen levels.** Blocks are the fix, and they are cheap.

![Before-and-after comparison: a single tensor-wide scale dominated by one outlier wastes levels, while a per-block scale lets each block of 32 weights use its full integer range](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-3.webp)

LLM weights are not uniformly distributed. The overwhelming majority cluster tightly around zero — standard deviations of $0.02$ to $0.05$ are typical — but a small number of weights, often in specific channels, are an order of magnitude larger. If you compute one scale for the whole tensor, that scale is hostage to the largest weight. With $d = \mathrm{max}/15$ driven by an outlier of $2.0$, the spacing is $0.13$, and a weight of $0.04$ rounds to the same level as a weight of $0.09$. You have effectively quantized the bulk of your model to one or two bits while spending your precision on a handful of giants.

Those giants are not noise; they are doing work. A well-documented property of trained transformers is that a small number of feature directions — often tied to specific channels in the down projection and to certain tokens such as delimiters and the beginning-of-sequence marker — carry outsized activation and weight magnitudes, and clipping them wrecks the model. This is the same outlier phenomenon that makes *activation* quantization hard enough to need dedicated methods like SmoothQuant and LLM.int8(). A quantizer cannot simply clip the tails and move on. The outliers must be represented; the only real question is how much of the bit budget they are allowed to consume, and whether they get to hijack the scale for their innocent neighbours.

The per-block fix is to recompute `d` for every 32 weights. Now the one block that contains the outlier pays the price of a coarse scale, and the other 127 blocks in a 4096-wide row each get a scale tuned to their own tiny range. The error, remember, is proportional to `d`, so shrinking `d` for 99% of the blocks shrinks the error almost everywhere. The cost is the metadata: one fp16 scale per 32 weights is $2 \times 8 / 32 = 0.5$ extra bits per weight. That half-bit is the best deal in the entire field.

You can measure the payoff directly:

```python
import numpy as np
rng = np.random.default_rng(0)

w = rng.normal(0, 0.05, size=4096).astype(np.float32)   # realistic small weights
w[::512] = rng.uniform(-2.0, 2.0, size=8)               # eight fat outliers

def rms_error(x, block, n_bits=4):
    qmax = (1 << n_bits) - 1
    x = x.reshape(-1, block)
    lo = x.min(1, keepdims=True)
    hi = x.max(1, keepdims=True)
    d = (hi - lo) / qmax
    q = np.clip(np.round((x - lo) / d), 0, qmax)
    xh = d * q + lo
    return float(np.sqrt(((xh - x) ** 2).mean()))

for block in (4096, 256, 32):
    print(f"block = {block:>5}   RMS error = {rms_error(w, block):.5f}")
```

Running it prints something close to:

```
block =  4096   RMS error = 0.07771
block =   256   RMS error = 0.01032
block =    32   RMS error = 0.00643
```

Going from one scale for the whole row (block 4096) to a scale every 32 weights cuts the error by roughly twelve times, for half a bit of overhead. This is not a marginal optimization; it is the reason 4-bit quantization is usable at all.

### Granularity has a name in every framework

The same idea shows up under different vocabulary depending on the stack. *Per-tensor* is one scale for the matrix (what naive INT8 does). *Per-channel* or *per-row* is one scale per output row (common in GPTQ-style methods). *Per-group* is one scale per group of, say, 128 weights along the input dimension (GPTQ's `group_size=128`). GGUF's legacy formats are per-block with block 32; its K-quants are a two-level scheme we will dissect shortly. If you have read about [INT8 versus INT4 edge tradeoffs](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs), this is the same granularity axis, viewed from the file format instead of the kernel.

### The second-order gotcha: smaller blocks are not free

It is tempting to conclude that tiny blocks are strictly better. They are not, for two reasons. First, the metadata overhead grows: a block of 8 with an fp16 scale spends 2 bits per weight on scale alone, which at 4-bit payload means a third of your bytes are bookkeeping. Second, an fp16 scale is itself a 16-bit number you have to store per block, and at some point storing thousands of full-precision scales is wasteful when those scales are themselves highly correlated. The K-quants exist precisely to resolve this tension — they keep the small block size for accuracy but quantize the scales themselves. That is the next section.

## 3. Decoding the legacy formats: q4_0, q4_1, q5_0, q5_1, q8_0

**The senior rule of thumb: read a legacy quant name as `q{bits}_{has_min}`. The number is the payload width; the digit is whether a minimum is stored.** Once you see that, the byte layout writes itself.

![Five byte-strip diagrams decoding q4_0, q4_1, q5_0, q5_1, and q8_0 block layouts, showing the scale, optional minimum, optional high-bit field, and quantized payload with byte counts and bits-per-weight](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-4.webp)

Every legacy format packs exactly 32 weights into one block. What changes is which fields sit in front of the packed integers. The figure lays out all five to scale; here is the same information as a table you can keep next to your terminal:

| Format | Fields per 32-weight block | Block bytes | Bits / weight | Reconstruction |
|---|---|---|---|---|
| `q4_0` | `d` (fp16) + 32x4-bit | 18 | 4.50 | $\hat{w} = d\,(q - 8)$ |
| `q4_1` | `d` (fp16) + `m` (fp16) + 32x4-bit | 20 | 5.00 | $\hat{w} = d\,q + m$ |
| `q5_0` | `d` (fp16) + 32x1-bit high + 32x4-bit low | 22 | 5.50 | $\hat{w} = d\,(q - 16)$ |
| `q5_1` | `d` (fp16) + `m` (fp16) + 32x1-bit + 32x4-bit | 24 | 6.00 | $\hat{w} = d\,q + m$ |
| `q8_0` | `d` (fp16) + 32x8-bit | 34 | 8.50 | $\hat{w} = d\,q$ |

Walk the rows. `q4_0` is the original 4-bit format: one fp16 scale, then 32 nibbles (4 bits each, $32 \times 4 / 8 = 16$ bytes), $2 + 16 = 18$ bytes, $18 \times 8 / 32 = 4.5$ bits per weight. The "0" tells you there is no minimum, so the levels are symmetric and the reconstruction subtracts the implicit offset of 8. `q4_1` adds a second fp16 field, the per-block minimum `m`, growing the block to 20 bytes and switching the reconstruction to the affine form $d\,q + m$. That is the whole difference between them: two bytes and an add.

The 5-bit formats expose a packing wrinkle worth seeing, because it explains why odd bit widths are rarer than even ones. You cannot store a 5-bit integer in a clean nibble, so `q5_0` splits each value: the low 4 bits go in a normal nibble array (`ql`, 16 bytes) and the 32 high bits are bit-packed into a separate 4-byte field (`qh`). The kernel reassembles `q = ql | (qh_bit << 4)` to recover a value in $[0, 31]$, then applies $\hat{w} = d\,(q - 16)$. `q5_1` does the same with an added `m`. This split-field trick reappears, scaled up, inside the K-quants — anytime you see a `qh` field, it is the overflow bits that did not fit in the main payload.

`q8_0` is the high-precision anchor: one fp16 scale and 32 signed 8-bit integers, no minimum, $\hat{w} = d\,q$. At 8.5 bits per weight it is within a rounding error of fp16 quality on every model anyone has measured, which makes it the reference you compare the lossy formats against. It is also the format the kernels quantize *activations* into for the integer dot product, which is why you will see `q8_0` and its cousin `q8_1` show up even in models stored at 4 bits.

### How the blocks are actually multiplied

It is worth being precise about what happens to these blocks at inference, because the popular phrase "dequantize on the fly" is slightly misleading. When `llama.cpp` multiplies a quantized weight matrix by an activation, it does not expand the weights to fp16 and call a float GEMM. Instead, every quantized type declares a companion *activation* type — its `vec_dot_type` — and the kernel quantizes the incoming activation row into that type once, then computes each output element as an integer dot product between a weight block and an activation block, scaling the integer accumulator by the product of the two blocks' scales and summing in float.

The companion types are exactly the `q8` formats. The legacy symmetric formats (`q4_0`, `q5_0`, `q8_0`) pair with `q8_0`. The affine formats (`q4_1`, `q5_1`) pair with `q8_1`, which is the reason `q8_1` exists at all: the affine reconstruction $d\,q + m$ produces a cross term proportional to the *sum* of the activations, and `q8_1` stores that per-block sum precomputed so the minimum's contribution can be folded in with one multiply instead of a loop. The K-quants pair with `q8_K`, the 8-bit super-block format. So a model "stored at 4 bits" is, in the inner loop, doing 4-bit-by-8-bit integer dot products, accumulating in int32, and rescaling — which maps directly onto the integer SIMD instructions on CPUs (AVX2, AVX-512 VNNI, ARM NEON dot-product) and the DP4A or tensor-core paths on GPUs.

Two consequences fall out of this. First, the activation quantization to 8 bits is a small additional error source on top of the weight quantization, usually negligible but real, and it is why even an otherwise-lossless `q8_0` weight is not bit-identical to fp16 inference. Second, the speed of a quant is not just its size — it is how well its `vec_dot` maps onto the target hardware's integer instructions. That is the mechanism behind the I-quant CPU decode penalty we will meet later: a uniform quantizer's `vec_dot` is a tight multiply-accumulate, while a codebook quantizer's involves a table gather, and gathers are slow on CPUs.

### What happened to q4_2 and q4_3

If you go digging through llama.cpp history you will find references to `q4_2` and `q4_3`, and they are gone. They were experiments with different block sizes and packings for 4-bit, made obsolete the moment the K-quants landed and delivered better quality at the same bit budget. This is the recurring story of the legacy formats: they are simple, fast, supported everywhere, and almost always the wrong choice today. You reach for `q4_0` or `q8_0` when you need a format that every downstream tool understands without question, or when you are on hardware where the K-quant kernels are not optimized. For quality per byte, the K-quants win, and have since 2023.

### The second-order gotcha: fp16 scales can overflow

The scale `d` is stored as fp16, whose maximum finite value is $65504$. For normal weights this is never close to a problem. But on a pathological block — say one weight at $40000$ from a numerical bug in a fine-tune, or an un-clipped embedding row — the computed scale can exceed fp16's range and saturate to infinity, which then poisons every weight in the block to `NaN` at the first matmul. The legacy formats have no defense here. The K-quants, by scaling their sub-block scales through a shared fp16 master, spread the dynamic range across two numbers and are far more robust to a single giant value. I have watched exactly this turn a model to garbage at load time; the fix was a higher-precision quant, not a different sampler.

## 4. K-quants: super-blocks and quantized scales

**The senior rule of thumb: a K-quant is a block-of-blocks. It keeps the accuracy of a small 32-weight scale but stops paying full fp16 price for each scale by quantizing the scales themselves to 6 bits under one shared master.** This is the single most important idea in modern GGUF, and once it clicks the entire `_K` family is obvious.

![Tree diagram of a Q4_K super-block: a root holding 256 weights in 144 bytes, branching to an fp16 master scale and master min, a 12-byte field of eight 6-bit sub-block scales, and 128 bytes of 4-bit quants](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-5.webp)

Recall the tension from section 2: small blocks give low error but their fp16 scales become expensive overhead. The K-quant resolution is a two-level hierarchy. Group 256 weights into a **super-block**. Inside it, keep eight **sub-blocks** of 32 weights each — the same granularity that gave us low error. But instead of storing eight fp16 scales (16 bytes), store eight *6-bit* scales (and eight 6-bit minimums), packed into 12 bytes, and add just two fp16 "master" values — `d` and `dmin` — for the whole super-block. The effective scale for sub-block $j$ is the product of the master and the 6-bit value:

$$
\text{scale}_j = d \cdot \mathrm{sc}_6[j], \qquad \text{min}_j = d_{\min} \cdot \mathrm{m}_6[j], \qquad \hat{w} = \text{scale}_j \cdot q - \text{min}_j
$$

So every weight is rebuilt as $\hat{w} = d \cdot \mathrm{sc}_6[j] \cdot q - d_{\min} \cdot \mathrm{m}_6[j]$, where $q$ is the 4-bit payload, $\mathrm{sc}_6[j]$ and $\mathrm{m}_6[j]$ are the 6-bit sub-block scale and minimum, and $d, d_{\min}$ are the two fp16 masters shared across all 256 weights. The figure traces exactly this: the root super-block fans out to `d`, `dmin`, the 12-byte scales field, and the 128-byte payload; the scales field fans out again to the per-sub-block 6-bit scale and minimum.

Let us count the bytes for `Q4_K`, the 4-bit member: two fp16 masters ($2 + 2 = 4$ bytes), the packed 6-bit scales and minimums (12 bytes), and 256 four-bit quants ($256 \times 4 / 8 = 128$ bytes). That is $4 + 12 + 128 = 144$ bytes for 256 weights, or $144 \times 8 / 256 = 4.5$ bits per weight — the same headline number as `q4_0`, but with eight independent sub-block scales instead of one, at a fraction of the scale overhead. That is why `Q4_K` beats `q4_0` at identical bits per weight: better scale granularity, paid for by quantizing the scales.

The whole family follows the same pattern with different payload widths and slightly different bookkeeping:

| Format | Payload bits | Super-block bytes (256 w) | Bits / weight | Where the bits go |
|---|---|---|---|---|
| `Q2_K` | 2 | 84 | 2.625 | 16 B scales + 64 B quants + 4 B masters |
| `Q3_K` | 3 | 110 | 3.4375 | 32 B high bits + 64 B low + 12 B scales + 2 B master |
| `Q4_K` | 4 | 144 | 4.50 | 4 B masters + 12 B scales + 128 B quants |
| `Q5_K` | 5 | 176 | 5.50 | 4 B masters + 12 B scales + 32 B high + 128 B low |
| `Q6_K` | 6 | 210 | 6.5625 | 128 B low + 64 B high + 16 B 8-bit scales + 2 B master |

A few details reward attention. `Q6_K` stores its sub-block scales as full 8-bit signed integers rather than 6-bit, because at six payload bits the scale precision starts to matter again. `Q2_K` and `Q3_K` use the split high-bit/low-bit packing we met in `q5_0`, because 2 and 3 do not divide a byte cleanly. And `Q8_K`, which you will rarely select but which appears internally, is the 8-bit super-block format used to quantize activations for the K-quant dot products.

Here is the two-level scheme in NumPy, both the encode and the byte accounting:

```python
import numpy as np

SUPER, SUB = 256, 32                                  # super-block, sub-block
w = np.random.default_rng(1).normal(0, 0.05, SUPER).astype(np.float32).reshape(8, SUB)

lo, hi = w.min(1), w.max(1)                            # per sub-block range
sub_scale = (hi - lo) / 15.0                           # 8 real fp32 scales
sub_min   = -lo                                        # 8 real fp32 minimums

d    = sub_scale.max() / 63.0                          # fp16 master for the scales
dmin = sub_min.max()   / 63.0                          # fp16 master for the minimums
sc6  = np.round(sub_scale / d).astype(np.int32)        # 6-bit scales,   0..63
m6   = np.round(sub_min   / dmin).astype(np.int32)     # 6-bit minimums, 0..63

eff_scale = (d * sc6)[:, None]                          # reconstruct effective scale
eff_min   = (dmin * m6)[:, None]
q  = np.clip(np.round((w + eff_min) / eff_scale), 0, 15)
wh = eff_scale * q - eff_min                            # dequantize:  d*sc*q - dmin*m

bytes_per_block = 2 + 2 + 12 + 128                      # d, dmin, scales, quants
print("bytes      =", bytes_per_block, "for", SUPER, "weights")
print("bits/weight=", round(bytes_per_block * 8 / SUPER, 4))   # 4.5
print("RMS error  =", round(float(np.sqrt(((wh - w) ** 2).mean())), 5))
```

### Reading the byte budget, and the high end of the family

The byte tables above repay a second look, because the packing is where the cleverness hides. In `Q4_K`, the 12-byte `scales` field is not eight scales and eight minimums laid out plainly. Eight 6-bit scales plus eight 6-bit minimums is $8 \times 6 + 8 \times 6 = 96$ bits, which is 12 bytes exactly — but only because the encoder bit-packs sixteen 6-bit values with no padding, splitting the high and low bits across the field. `Q6_K` makes a different trade: at six payload bits the sub-block scales need more than 6 bits of precision themselves, so it stores them as full signed 8-bit integers (a 16-byte `scales` field for 16 sub-blocks of 16 weights) and drops the separate minimum, reconstructing as $\hat{w} = d \cdot \mathrm{sc}_8[j] \cdot q$. That is why `Q6_K` is symmetric where `Q4_K` is affine, and why it lands at 6.5625 bits per weight rather than a round 6.

`Q5_K` and `Q6_K` are the high end you reach for when `Q4_K_M` is not enough — structured output, code, or long-context tasks where a small per-token error compounds over thousands of tokens into a derailed generation. The jump from `Q4_K_M` to `Q5_K_M` is the last clearly worthwhile step on the quality curve; `Q6_K` is for when you want fp16-class behavior and can afford 6.5 bits. Above that, `Q8_0` exists mostly so you have a number to compare against. The corresponding GGML type enums — `GGML_TYPE_Q4_K`, `GGML_TYPE_Q6_K`, and so on — are what you will see in the source and in a `gguf_dump`, and they map one-to-one onto these structs in `ggml-quants.h`. If you only ever remember one thing about K-quants, make it this: the scales are quantized too, and that recursion is the whole trick.

### The second-order gotcha: why Q3_K_M can beat Q4_0

People are surprised that a 3.44-bit `Q3_K_M` often scores closer to fp16 than a 4.5-bit `q4_0`, despite carrying a full bit less. The reason is everything in this section: `Q3_K` has eight finely-tuned sub-block scales where `q4_0` has one coarse scale, and the K-quant recipes additionally route the sensitive tensors to higher precision (next section). The bit budget on the tin is not the quality. *How* the bits are spent — on scale granularity, on the right tensors, on an importance-weighted fit — is the quality. This is the single most common misconception I correct when someone is choosing a quant.

## 5. The _S / _M / _L suffix is a recipe, not a format

**The senior rule of thumb: `Q4_K_M` is not a block format. It is a per-tensor type-assignment policy that happens to use `Q4_K` for most tensors and `Q6_K` for a handful of sensitive ones.** The suffix `_S`, `_M`, `_L` tells you how aggressive that upgrade policy is, not how the bytes are laid out.

![Matrix comparing Q4_K_S and Q4_K_M across tensor roles, showing both store Q4_K for most tensors but Q4_K_M upgrades the value projection, the down projection, and the output head to Q6_K](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-6.webp)

This is the fact that, once internalized, changes how you read every GGUF filename. A transformer layer is not one undifferentiated blob of weights. It has query, key, and value projections; an attention output projection; gate, up, and down projections in the feed-forward block. On top of the stack sit the token embedding and the output head (the final projection to logits). These tensors are *not equally sensitive* to quantization error. The value projection and the feed-forward down projection sit on the residual stream's critical path, and the output head turns hidden states directly into logits — an error there corrupts the probability of every token. Crushing those to 4 bits costs disproportionately more quality than crushing the query projection.

So the K-quant recipes do not apply one format uniformly. They assign types per tensor. The figure shows the policy for the two most common 4-bit recipes. `Q4_K_S` ("small") is nearly uniform: `Q4_K` for essentially everything, with the output head sometimes nudged up. `Q4_K_M` ("medium") keeps `Q4_K` for the query, key, attention-output, gate, and up projections, but promotes the **value projection**, the **feed-forward down projection**, and the **output head** to `Q6_K`. Those three upgrades are why `Q4_K_M` measurably beats `Q4_K_S` for a small increase in file size, and why its *effective* bits per weight is around 4.8, not the 4.5 of pure `Q4_K`.

This per-tensor logic lives in `llama.cpp`'s `llama_tensor_get_type` function, and the exact layer selection is version- and architecture-dependent — newer versions tweak which layers and how many get the upgrade, and mixture-of-experts models like the [Qwen and DeepSeek architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) have their own rules for the expert tensors. Do not memorize specific layer indices; memorize the *shape* of the policy: most tensors at the nominal width, a small set of sensitive tensors bumped up, the output head almost always protected.

You can see the actual assignment for any file. After quantizing, dump the tensor types:

```bash
python convert_hf_to_gguf.py ./Meta-Llama-3-8B --outfile llama3-8b-f16.gguf --outtype f16
./llama-quantize llama3-8b-f16.gguf llama3-8b-Q4_K_M.gguf Q4_K_M
python gguf-py/scripts/gguf_dump.py --no-tensor llama3-8b-Q4_K_M.gguf \
  | grep -E "attn_v|ffn_down|output|token_embd"
```

You will see lines reporting `Q6_K` next to `attn_v.weight`, `ffn_down.weight`, and `output.weight`, and `Q4_K` next to the rest. The filename said "Q4_K_M"; the bytes say "mostly Q4_K, strategically Q6_K." That gap is the recipe.

One tensor deserves special mention because people forget it counts: the token embedding. On a large-vocabulary model it can be hundreds of millions of parameters, so its quantization affects file size noticeably, and because it is looked up rather than multiplied, its errors behave differently — a coarsely quantized embedding row degrades one specific token's representation everywhere that token appears, which is exactly how you get a model that is fluent except when it hits a particular rare word. Most recipes keep it at `Q4_K` and bump it for the larger suffixes. Mixture-of-experts models complicate the picture further, because the expert feed-forward tensors dominate the parameter count and carry their own promotion rules — quantizing all experts uniformly versus protecting the shared/router weights is a live design choice in those recipes. The point stands: the recipe is a per-tensor decision, and the interesting tensors are the embedding, the output head, and whatever sits on the residual stream's critical path.

### The second-order gotcha: _L, and when the suffix lies

`_L` ("large") pushes the upgrade further — more tensors to `Q6_K`, sometimes the embedding too — and on some recipes the marginal quality is not worth the marginal bytes; you would often be better served by the next width up (`Q5_K_M`) than by `Q4_K_L`. Worse, because the suffix is a policy and not a format, two files both named `Q4_K_M` produced by different llama.cpp versions can have genuinely different tensor assignments and therefore different quality and size. When you are comparing quants rigorously — say, for a benchmark — never trust the filename alone. Dump the types. The recipe is the ground truth.

## 6. I-quants: codebooks and the importance matrix

**The senior rule of thumb: an I-quant abandons the evenly-spaced ruler. Instead of levels at regular intervals, it spends a small shared codebook of irregular points on the dense centre of the weight distribution, and uses an importance matrix to decide which weights deserve the precision.** This is what lets `IQ2` and `IQ3` stay coherent where `Q2_K` falls apart.

![Before-and-after comparison of K-quant uniform grid versus I-quant codebook: uniform levels waste codepoints in the distribution tails, while a shared non-uniform lattice codebook fit with an importance matrix achieves lower error at low bit widths](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-7.webp)

Everything so far has used a *uniform* quantizer: pick a scale, lay down equally-spaced ticks, snap. That is optimal only if weights are uniformly distributed inside a block, which they are not — they are roughly Gaussian, dense near zero and sparse in the tails. A uniform grid spends as many of its precious levels on the empty tails as on the crowded centre. At 4 or more bits you have enough levels that this waste is tolerable. At 2 bits you have *four levels total*, and spending even one of them on a tail you rarely visit is catastrophic. This is why `Q2_K` produces fluent gibberish: it simply does not have the levels to spare for a uniform grid.

The I-quants ("I" for the importance-matrix lineage) take a different tack borrowed from classical vector quantization. They define a fixed **codebook** — a set of points, often arranged on a lattice such as the E8 lattice, chosen offline to match the shape of a normalized weight distribution. Each group of weights is then represented not by per-weight integers but by an *index* into this shared codebook, which maps to a short vector of reconstructed values. Because the codebook points are placed where weights actually live — clustered near zero, sparse in the tails — you get dramatically lower error per bit than a uniform grid in the very-low-bit regime. The cost is on the decode side: looking up codebook entries is not the clean shift-and-multiply of a uniform quantizer, so I-quants decode meaningfully slower, especially on CPUs where the lookups defeat SIMD.

The I-quant lineup, by nominal bits per weight:

| Format | Bits / weight | Typical use |
|---|---|---|
| `IQ1_S` / `IQ1_M` | 1.56 / 1.75 | Extreme: a huge model that otherwise will not load at all |
| `IQ2_XXS` / `IQ2_XS` | 2.06 / 2.31 | Run a 70B-class model on 24 GB of VRAM |
| `IQ2_S` / `IQ2_M` | 2.50 / 2.70 | The sweet spot of the 2-bit range; beats `Q2_K` clearly |
| `IQ3_XXS` / `IQ3_S` / `IQ3_M` | 3.06 / 3.44 / 3.66 | Squeeze a bigger model than `Q4_K_M` would allow |
| `IQ4_XS` / `IQ4_NL` | 4.25 / 4.50 | Non-linear 4-bit; a touch better than `Q4_K` per byte |

### What is actually in the codebook

The codebook is not learned per model; it is a fixed table baked into `llama.cpp`. For the 4-bit non-linear format `IQ4_NL`, the table is sixteen carefully chosen real values — not evenly spaced, but bunched near zero and spread out in the tails to match a normalized weight distribution — and quantizing a weight means finding the nearest of those sixteen points and storing its 4-bit index. The lower-bit I-quants go further and encode *groups* of weights at once using a lattice: rather than one index per weight, a single index selects a short vector of reconstructed values from a structured set of points, with the codebooks derived from lattices such as $E_8$ that pack points efficiently in high-dimensional space. Encoding several weights jointly is what lets `IQ2` and `IQ3` drop below the per-weight bit budget a uniform quantizer could reach — the information is amortized across the group, and the most common group patterns get the shortest codes.

This joint encoding is also the source of the decode cost. A uniform quantizer rebuilds a weight with a shift and a multiply. A codebook quantizer rebuilds a group by reading a table at a computed index — a memory gather. On a GPU with the optimized kernels the gather hides behind memory latency; on a CPU it stalls the pipeline. The format that scores best on perplexity per byte is therefore not automatically the fastest, and which one wins depends entirely on where you run it. The `IQ1_S` and `IQ1_M` formats push this to roughly one and three-quarter bits per weight by leaning hardest on the shared codebook, and they only stay coherent at all on very large models with a good importance matrix — which is the next piece.

### The importance matrix

The codebook fixes *where* the levels are. The **importance matrix** (imatrix) fixes *which weights get the accuracy*. It is a per-column importance weight, computed by running a few hundred kilobytes of representative text through the full-precision model and accumulating, for each weight column, how much that column's activations contribute to the output. Columns that fire hard and often on real data are weighted heavily; the quantizer is then told to minimize error on those columns preferentially, even at the expense of columns that barely matter. The objective shifts from "minimize raw weight error" to "minimize error *that the model will actually feel*."

For I-quants the imatrix is not optional in practice — the low-bit codebook fits are far worse without it. For K-quants it is an optional but usually worthwhile improvement, especially at `Q3_K` and below. Generating one is a single command:

```bash
./llama-imatrix -m llama3-8b-f16.gguf -f calibration.txt -o llama3-8b.imatrix --chunks 200
./llama-quantize --imatrix llama3-8b.imatrix llama3-8b-f16.gguf llama3-8b-IQ3_M.gguf IQ3_M
```

The one decision that matters here is the calibration text. It must look like what the model will see in production. A general-purpose imatrix is usually built from a broad text sample (wiki, code, a little multilingual), and that is fine for a general assistant. But if you are quantizing a specialized model — a code model, a model for a non-English language, a domain model — and you calibrate on generic English prose, you will protect the wrong columns and the model will degrade exactly on the task you care about. This is the most common self-inflicted wound with I-quants, and we will see it in the case studies.

### The second-order gotcha: the CPU decode cliff

I-quants are a GPU-first technology. On a card with the optimized kernels, the slower decode is hidden behind memory bandwidth and you get the quality win nearly for free. On a CPU-only box — a laptop, a cheap VPS — the codebook lookups dominate and an `IQ4_XS` can decode noticeably slower than a `Q4_K_M` of similar size, sometimes by a third or more. If you are CPU-bound, the uniform K-quants are usually the better engineering choice even though the I-quant scores higher on perplexity. Always benchmark decode tokens-per-second on your actual hardware, not just the quality metric.

## 7. Placing GGUF in the wider quantization taxonomy

**The senior rule of thumb: GGUF is one leaf of a large tree. Knowing which leaf tells you what it is good at and what it structurally cannot do.** It is post-training, weight-only, static, and per-block. Each of those words excludes a different alternative.

![Tree diagram placing GGUF in the quantization taxonomy: post-training versus train-aware, then weight-only versus weight-plus-activation, with GGUF as the weight-only per-block static leaf alongside GPTQ, AWQ, SmoothQuant, and QLoRA](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-8.webp)

The four axes that locate GGUF:

**Post-training (PTQ) versus training-aware (QAT).** GGUF quantizes a finished model with no gradient steps — you take fp16 weights, run the recipe, ship. Training-aware quantization instead simulates the rounding during training so the network learns weights that survive it, which is what QLoRA-style fine-tuning and native low-bit training do. PTQ is cheap and good enough above 3 bits; QAT is expensive but the only way to hold quality at the extreme low end. The frontier of that low end — rotation methods, learned codebooks, native ternary training — is its own deep topic, covered in [past the 4-bit wall](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization).

**Weight-only (W4A16) versus weight-plus-activation (W8A8).** GGUF quantizes weights and leaves activations in higher precision (it quantizes them to 8-bit transiently for the dot product, but they are not stored quantized). This is what you want for the memory-bandwidth win on decoding. Weight-plus-activation schemes like SmoothQuant and LLM.int8() quantize both so the matmul itself runs on INT8 tensor cores — a compute win that matters for prefill-heavy, high-batch serving, at the cost of the harder activation-outlier problem. The two families optimize different bottlenecks.

**Static versus dynamic.** GGUF's scales are computed once at quantization time and frozen — static. Some schemes compute activation scales on the fly per forward pass — dynamic — trading a little runtime cost for adaptivity. Weight quantization is naturally static because weights do not change; the dynamic question only arises for activations.

**Per-block granularity**, which we have already dissected. GGUF lives at the fine-grained end, which is a large part of why it holds quality well.

Here is the same map as a table, with the neighbours GGUF is most often confused with:

| Method | Paradigm | What it quantizes | Granularity | Lives where |
|---|---|---|---|---|
| GGUF (q/K/IQ) | PTQ | weights only | per-block / super-block | llama.cpp, on CPU and GPU |
| GPTQ | PTQ | weights only | per-group, Hessian-aware | GPU serving (vLLM, TGI) |
| AWQ | PTQ | weights only | per-group, activation-aware | GPU serving |
| SmoothQuant | PTQ | weights + activations | per-channel | INT8 GPU inference |
| bitsandbytes NF4 | PTQ | weights only | per-block, normal-float | Hugging Face / QLoRA base |
| QLoRA | QAT-adjacent | weights (NF4) + adapters | per-block | fine-tuning |

One structural difference is worth calling out, because it surprises people moving between ecosystems. GGUF stores weights in these block layouts and reconstructs them inside the matmul kernel, which is what lets the same `.gguf` file run on a CPU, a Mac's Metal backend, or an NVIDIA GPU with no repacking. GPTQ and AWQ instead pre-pack their 4-bit weights into a layout tuned for a specific GPU kernel — Marlin, Machete, ExLlama — trading portability for raw throughput on that hardware. Neither is wrong; they optimize for different deployment surfaces. If your target is "whatever hardware the user happens to have," GGUF's reconstruct-in-kernel design is the reason it runs everywhere. If your target is a fleet of identical datacenter GPUs, the pre-packed formats extract more tokens per second from them.

The practical upshot: GGUF is the format you reach for when the deployment target is `llama.cpp` and its ecosystem — desktop apps, single-GPU boxes, CPUs, edge devices, anything running a `.gguf` file. For multi-GPU production serving you are more likely on GPTQ or AWQ inside vLLM. They are not competitors so much as different doors into the same building; the underlying math — scales, blocks, the outlier problem — is shared, as our [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) discussion lays out.

## 8. Bits per weight versus quality: the real frontier

**The senior rule of thumb: quality as a function of bits is a hockey stick. It rises steeply from 2 to 4 bits, bends hard around 4 to 5, and is nearly flat above 6. Spend your bytes on the steep part of the curve.** `Q4_K_M` sits right at the bend, which is why it is the default for a reason, not just by habit.

![Matrix laddering quant formats from Q2_K to Q8_0 by bits per weight, with columns for a 7B model's on-disk size, typical perplexity increase versus fp16, and a one-line verdict, highlighting Q4_K_M as the sweet spot](/imgs/blogs/how-quantization-works-gguf-quant-types-decoded-9.webp)

The figure is the ladder. The perplexity deltas are illustrative — the exact numbers depend on the model, the dataset, and whether an imatrix was used — but the *shape* is robust across every model family I have measured: the jump from `Q2_K` to `Q3_K_M` recovers a large fraction of the lost quality, `Q3_K_M` to `Q4_K_M` recovers most of the rest, and from `Q5` upward you are paying real bytes for differences you need a careful benchmark to even detect.

| Format | Bits / weight | 7B on disk | Typical PPL increase vs fp16 | Verdict |
|---|---|---|---|---|
| `Q2_K` | 2.63 | ~2.7 GB | large (+0.8 to +1.5) | last resort; prefer `IQ2_M` at this size |
| `IQ3_XXS` | 3.06 | ~2.9 GB | moderate (+0.4) | big model, tiny VRAM |
| `Q3_K_M` | 3.44 | ~3.3 GB | small (+0.25) | tight fit, still coherent |
| `Q4_0` | 4.50 | ~3.8 GB | small (+0.12) | legacy; use `Q4_K_M` instead |
| `Q4_K_M` | 4.83 (effective) | ~4.1 GB | very small (+0.06) | the default sweet spot |
| `Q5_K_M` | 5.67 (effective) | ~4.8 GB | tiny (+0.03) | extra headroom for sensitive work |
| `Q6_K` | 6.56 | ~5.5 GB | negligible (+0.01) | near-lossless without paying for 8 bits |
| `Q8_0` | 8.50 | ~7.2 GB | ~0 | reference / baseline only |

Two practical readings of this table. First, `Q8_0` is almost never the right production choice — it doubles your bytes versus `Q4_K_M` to chase a quality difference that is below the noise floor of most evaluations. Keep it for establishing a baseline, not for shipping. Second, the interesting decisions all happen below 4 bits, where the format *family* (K-quant versus I-quant) matters as much as the bit count, and where the importance matrix earns its keep.

It also pays to know that the curve shifts with model scale. Larger models are more robust to quantization — a 70B at `Q3_K_M` typically feels better than a 7B at the same recipe, because the redundancy quantization eats into is more plentiful. This is why the extreme low-bit I-quants are most compelling on the biggest models: a 70B at `IQ2_M` can beat a 13B at `Q5_K_M` of similar file size on many tasks, because parameters spent on a bigger architecture, even crushed to two bits, can outvalue fewer parameters kept precise. The corollary is that small models punish aggressive quantization hardest — a 1B or 3B model at `Q3` or below degrades fast, and you are usually better off at `Q5_K_M` or `Q6_K` and accepting the size. The right bit width is not a constant; it is a function of how many parameters you started with.

### The full workflow, end to end

Putting the whole pipeline together, here is how you go from Hugging Face weights to a tuned quant and verify what you got:

```bash
python convert_hf_to_gguf.py ./Meta-Llama-3-8B --outfile llama3-8b-f16.gguf --outtype f16
./llama-imatrix -m llama3-8b-f16.gguf -f calibration.txt -o llama3-8b.imatrix --chunks 200
./llama-quantize --imatrix llama3-8b.imatrix llama3-8b-f16.gguf llama3-8b-Q4_K_M.gguf Q4_K_M
./llama-perplexity -m llama3-8b-Q4_K_M.gguf -f wikitext-2-raw/wiki.test.raw
python gguf-py/scripts/gguf_dump.py --no-tensor llama3-8b-Q4_K_M.gguf | grep -E "type|Q6_K"
```

Convert to a full-precision GGUF, build an imatrix from representative text, quantize with the recipe you want, measure perplexity against a held-out set, and dump the tensor types to confirm the recipe did what you expected. If you change one variable — the recipe, the imatrix corpus, the source precision — re-measure. The whole reason to understand the formats is so that when perplexity moves, you know which knob moved it.

### The second-order gotcha: perplexity is not the only metric

A trap in low-bit quantization is to optimize perplexity and ship. Perplexity is an average over next-token predictions on generic text, and it is *weakly* correlated with the things users actually notice: instruction following, code correctness, long-context retrieval, refusing to hallucinate. A quant can hold perplexity while losing the ability to track a variable across 8000 tokens, because the rare, high-stakes tokens that long reasoning depends on are exactly the ones an aggressive quant rounds away. When you go below `Q4_K_M`, measure the task you care about, not just the proxy — the [conversational evaluation beyond perplexity](/blog/machine-learning/large-language-model/evaluating-conversational-llms-beyond-perplexity) post is the right companion here.

## Tracing one Q4_K weight, by hand

To make the two-level scheme fully concrete, let us decode a single weight from a `Q4_K` super-block the way the kernel does, with nothing hidden. Suppose we have read a super-block off disk and the relevant numbers are: master scale $d = 0.0123$, master minimum $d_{\min} = 0.0098$, and for sub-block $j = 3$ the stored 6-bit scale $\mathrm{sc}_6[3] = 47$ and 6-bit minimum $\mathrm{m}_6[3] = 12$. We want weight number 100 in the super-block, which falls in sub-block 3, and its stored 4-bit payload is $q = 9$.

Step one, reconstruct the effective scale and minimum for this sub-block from the masters:

$$
\text{scale}_3 = d \cdot \mathrm{sc}_6[3] = 0.0123 \times 47 = 0.5781, \qquad
\text{min}_3 = d_{\min} \cdot \mathrm{m}_6[3] = 0.0098 \times 12 = 0.1176
$$

Step two, apply the reconstruction $\hat{w} = \text{scale}_3 \cdot q - \text{min}_3$:

$$
\hat{w} = 0.5781 \times 9 - 0.1176 = 5.2029 - 0.1176 = 5.085
$$

That is the value this weight contributes to the matmul. Three things are worth noticing in this trace. First, the only per-weight storage was the single 4-bit number $9$ — everything else (the two masters, the two 6-bit sub-block values) is shared across 32 or 256 weights, which is why the amortized cost is 4.5 bits and not more. Second, the arithmetic is two multiplies and a subtract, all on small integers and two fp16 scalars, which is why dequantization is nearly free next to the memory traffic. Third, the error in $\hat{w}$ has two stacked sources: the 4-bit rounding of the weight itself, and the 6-bit rounding of the sub-block scale and minimum. The K-quant design bets that quantizing the scales to 6 bits costs less quality than it saves in bytes — and on real weight distributions, that bet pays off handsomely. If you internalize this trace, you can decode any `_K` format: only the bit widths of $q$, $\mathrm{sc}$, and $\mathrm{m}$ change.

## Case studies from production

Theory is cheap. These are the failures and fixes that teach the formats faster than any table.

### 1. The Q2_K lobotomy

A team wanted a 13B model on a 6 GB laptop GPU and reached for `Q2_K` because it was the only thing that fit. The model loaded, generated fluent English, and passed casual chat. Then it started failing every multi-step task: arithmetic with carries, following a three-clause instruction, anything requiring it to hold an intermediate result. Perplexity on wiki text was "only" up half a point, which had falsely reassured them. The root cause was the uniform 2-bit grid — four levels per sub-block is simply not enough resolution for the weights that route multi-step reasoning, and the errors compound across layers. The fix was to swap `Q2_K` for `IQ2_M` at almost the same file size: the non-uniform codebook plus an imatrix recovered the lost coherence, and the team kept their 13B on the 6 GB card. The lesson: in the 2-bit regime, the format family matters more than the model size you can brag about.

### 2. q4_0 versus the merged LoRA

An engineer fine-tuned a base model with LoRA, merged the adapter back into the weights, and quantized the result to `q4_0` for distribution — the simplest, most universally compatible format. The quantized model had a subtle but persistent degradation the base model did not. The cause was the merge: adding a low-rank update shifted several weight blocks into lopsided, distinctly non-zero-centred distributions. `q4_0`, being symmetric, forced its sixteen levels to straddle zero and wasted half of them on a range those blocks never occupied, doubling the effective error there. Switching to `q4_1` — which stores a per-block minimum and fits the actual range — closed most of the gap, and `Q4_K_M` closed the rest. The lesson: the symmetric/affine digit is not cosmetic, and post-merge weights are exactly where it bites.

### 3. The imatrix from the wrong corpus

A shop maintained a strong open-weights code model and wanted an `IQ3_M` build for constrained deployments. They generated the importance matrix from the same generic English text file they used for their chat models. The resulting quant wrote plausible prose about code but produced syntactically broken code — mismatched brackets, wrong indentation, hallucinated APIs — far more often than the fp16 original. The imatrix had weighted the columns that matter for English fluency and under-weighted the columns that fire on code tokens, so the aggressive 3-bit fit protected the wrong weights. Recomputing the imatrix on a few hundred kilobytes of actual source code, across the languages they cared about, restored the code quality. The lesson: an imatrix is only as representative as its calibration data, and "representative" is task-specific.

### 4. The output tensor that saved the rare tokens

A model quantized with an early, naive recipe that put *every* tensor at 4 bits — including the token embedding and the output head — developed a strange symptom: it was fine on common words but garbled rare tokens, proper nouns, and code identifiers, sometimes emitting near-misses like a transposed BPE merge. The output head maps hidden states directly to logits over the whole vocabulary, and 4-bit error there is enough to flip the argmax between two similar rare tokens whose logits are close. Switching to a `Q4_K_M` build, whose recipe keeps the output head at `Q6_K`, fixed it immediately. The lesson: not all tensors are equal, and the output head is the one you protect first — which is exactly the policy the `_M` suffix encodes.

### 5. The CPU I-quant cliff

A backend team benchmarked `IQ4_XS` against `Q4_K_M` on quality and chose `IQ4_XS` — it scored slightly better on perplexity at a slightly smaller size. In production on CPU-only inference nodes, decode throughput dropped by roughly a third versus their previous `Q4_K_M` build, blowing their latency budget. The cause was the I-quant decode path: codebook lookups do not vectorize the way a uniform quantizer's shift-and-multiply does, and on CPU that cost is exposed rather than hidden behind GPU memory bandwidth. They reverted to `Q4_K_M`, accepting the microscopic quality difference for the throughput. The lesson: pick the quant for the hardware you deploy on, and always benchmark tokens-per-second, not just the quality score.

### 6. The fp16 scale that saturated

A model quantized to `q8_0` produced `NaN` logits on the very first token for one specific prompt, only on one machine. The culprit was a single corrupted embedding row from an upstream conversion bug, with a weight near $50000$. The block's fp16 scale `d`, computed from that giant, exceeded fp16's $65504$ ceiling on the way through an intermediate and saturated to infinity, poisoning the whole block. Because `q8_0` stores one fp16 scale per block with no master to spread the dynamic range, it had no defense. Re-converting from clean source weights removed the corrupt value; as a belt-and-suspenders measure they also moved to a K-quant, whose two-level scaling is far more robust to a single extreme value. The lesson: fp16 scales have a finite range, and the legacy formats are brittle to pathological weights in a way the K-quants are not.

### 7. The 70B that only fit at IQ2_M

A researcher needed a 70B model on a single 24 GB consumer card for a latency-sensitive demo. At 2-ish bits per weight, the choices were `Q2_K` (2.63 bpw) and `IQ2_M` (2.70 bpw) — nearly identical file sizes, both just barely fitting with room for the KV cache. `Q2_K` produced a model that rambled and lost the thread within a few sentences. `IQ2_M`, with an imatrix built from the demo's actual prompts, stayed coherent through multi-paragraph answers and held the demo together. The two formats differ by a hundredth of a bit on paper and a world of usability in practice, entirely because of the non-uniform codebook and the importance-weighted fit. The lesson: at the bottom of the ladder, the I-quants are not a minor refinement — they are the difference between a usable model and a toy. If you push into KV-cache territory too, remember that the cache has its own quantization axis, discussed in [KV cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

### 8. The long-context model that forgot the middle

A retrieval-augmented assistant worked beautifully in testing on short prompts and fell apart in production on long ones — it answered using the start and end of a 16k-token context and silently ignored the middle. The team had quantized aggressively on two axes at once: `Q3_K_M` weights and a 4-bit KV cache to fit the window. Each was tolerable alone; together they pushed the model past the point where it could reliably attend to a fact buried mid-context. Perplexity looked fine because perplexity is dominated by short-range prediction and barely touches long-range retrieval. The fix was to spend bytes where they mattered: keep the weights at `Q4_K_M` and raise the KV cache to 8-bit, accepting a smaller maximum context in exchange for actually using the context it had. The lesson: quantization error on weights and on the KV cache are separate budgets that add up, and long-context behavior is the first thing to break and the last thing perplexity will warn you about.

### 9. Two files named Q4_K_M, two different models

A benchmarking effort compared a vendor's `Q4_K_M` release against an in-house `Q4_K_M` build and found a quality gap they could not explain — same base model, same nominal format, same size to within a few megabytes. Dumping the tensor types resolved it: the two files had been produced by `llama.cpp` versions a year apart, the per-tensor recipe had changed between them — different layers promoted to `Q6_K`, a different choice for the token embedding — and one had been built with an importance matrix while the other had not. Both were legitimately "Q4_K_M"; the suffix is a policy, and the policy had evolved. Once they regenerated both from the same version with the same imatrix, the gap closed. The lesson, again: in any rigorous comparison the filename is a label, not a specification. Dump the types and record the toolchain version, or you are comparing two different things and calling them the same.

## Which quant, when

The decoder ring is only useful if it ends in a decision. Here is the policy I actually follow.

**Reach for a given quant when:**

- **`Q4_K_M`** — your default for anything 7B and up that fits. It sits at the bend of the quality curve and protects the sensitive tensors. If you do not have a specific reason to deviate, this is the answer.
- **`Q5_K_M` or `Q6_K`** — when you have memory to spare and the task is precision-sensitive: code generation, structured output, long-context reasoning. `Q6_K` is near-lossless without paying the full 8-bit tax.
- **`Q8_0`** — only to establish a fp16-equivalent baseline for comparison, or when a downstream tool demands maximum compatibility and you do not care about size.
- **`Q3_K_M`** — when `Q4_K_M` does not quite fit and you would rather keep the bigger model than drop to a smaller one. Build an imatrix first.
- **`IQ3` / `IQ2` family** — when you are squeezing a much larger model into limited VRAM and you are on a GPU. Always with an imatrix calibrated on your real task. These beat the same-size K-quants clearly below 3 bits.
- **`q4_0` / `q8_0`** — when maximum tool compatibility matters more than quality per byte, or you are on hardware without optimized K-quant kernels.
- **`q4_1` over `q4_0`** — when your weights are not zero-centred, classically after a LoRA merge or in heavily fine-tuned models.

**Skip a quant when:**

- **`Q2_K`** for anything you care about — `IQ2_M` is the same size and meaningfully better; reserve uniform 2-bit for throwaway experiments.
- **I-quants on a CPU-only target** — the decode cliff usually outweighs the quality gain; benchmark before committing.
- **`Q8_0` in production** — you are doubling bytes for a quality difference below your evaluation's noise floor; that memory buys more in a bigger model or a longer context.
- **Any quant without an imatrix below `Q4_K_M`** — the importance matrix is most of the quality at low bits, and skipping it is leaving accuracy on the table for free.
- **Trusting a filename in a rigorous comparison** — dump the tensor types; two `Q4_K_M` files from different versions are not guaranteed identical.

As a starting point before you measure, this is the size-to-quant map I hand people, assuming a GPU target with room for a reasonable KV cache:

| Model fits comfortably? | VRAM headroom | First choice | If it does not fit |
|---|---|---|---|
| Yes, easily | model is half your VRAM | `Q6_K` or `Q5_K_M` | — |
| Yes | model is ~70% of VRAM | `Q4_K_M` | `Q4_K_S` |
| Barely | model is ~90% of VRAM | `Q4_K_S` or `Q3_K_M` (with imatrix) | `IQ3_M` |
| No, it is too big | a much larger model | `IQ3_XXS` / `IQ2_M` (with imatrix) | drop to a smaller model |
| CPU-only target | any | `Q4_K_M` (avoid I-quants) | `Q3_K_M` |

Treat it as a prior, not a verdict: build the candidate, measure perplexity and at least one real task, and adjust. The map gets you to the right neighbourhood; your evaluation picks the house.

The thread running through all of it: the name is a specification. `q4_0` is eighteen bytes and a subtraction. `Q4_K_M` is a super-block format plus a per-tensor upgrade policy. `IQ3_M` is a shared codebook fit with an importance matrix. Read the name as the spec it is, and the right choice for your memory budget, your hardware, and your task stops being folklore and becomes arithmetic.

## Further reading

- [Quantization in LLMs: the complete practical guide](/blog/machine-learning/large-language-model/quantization-in-llm) — the companion survey of methods (GPTQ, AWQ, GGUF, bitsandbytes) and the PTQ/QAT theory.
- [Past the 4-bit wall](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization) — rotation methods, vector codebooks, native ternary, and the kernels behind sub-4-bit quantization.
- [INT8 vs FP16 vs INT4: edge tradeoffs](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) — the same numeric formats viewed from the deployment and kernel side.
- [Efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) — why memory bandwidth, not compute, is what weight-only quantization actually buys you.
- The `llama.cpp` source: `ggml-quants.h` for the block structs, `ggml-quants.c` for the dequantize kernels, and `llama_tensor_get_type` in `llama.cpp` for the per-tensor recipes.
