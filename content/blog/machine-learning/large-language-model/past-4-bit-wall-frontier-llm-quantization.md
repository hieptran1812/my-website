---
title: "Past the 4-Bit Wall: The Frontier of LLM Quantization in 2026"
date: "2026-06-05"
publishDate: "2026-06-05"
description: "A staff-level tour of what happens to LLM quantization below 4 bits: rotation methods, vector codebooks, native ternary training, FP4 microscaling on Blackwell, and the kernels that decide whether any of it is actually fast."
tags: ["llm", "quantization", "inference", "optimization", "low-bit", "quarot", "spinquant", "aqlm", "bitnet", "nvfp4", "mxfp4", "kernels"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 52
aiGenerated: true
---

## Why the frontier moved past 4 bits

If you learned LLM quantization any time before about 2024, you learned a recipe that looks finished: take the FP16 weights, compute a scale per group of 128, round to INT4 with GPTQ or AWQ, ship it, enjoy the 3.5x memory cut. That recipe *is* finished — for 4-bit weight-only quantization on a GPU. It is a solved problem, well-supported by every serving stack, and I reach for it constantly. If that is what you need, you do not need this post; you need the [complete practical guide to quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm), which covers the math and the mainstream methods end to end.

This post is about everything that breaks the moment you try to go *further* — and about why teams keep trying to go further despite the breakage. The pressure is relentless and arithmetic: a 70B model in FP16 is 140 GB of weights. At 4-bit weight-only it is roughly 35 GB, which fits one 40 GB card with no room for a useful KV cache. At 2-bit it is roughly 18 GB, which fits a 24 GB consumer card *with* context. A native 1.58-bit model is roughly 13 GB and turns the dominant matrix multiply into integer additions. Every halving of the bit-width either unlocks a cheaper class of hardware or doubles the batch you can serve on the hardware you already own. The economic gradient points straight down, into bit-widths where the comfortable recipe simply stops working.

![The bit-width ladder from BF16 down to ternary, and the method that unlocks each rung](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-1.webp)

The diagram above is the mental model for the whole article: each rung down the ladder is unlocked by a *different class of technique*, not by turning a knob on the previous one. From 16 to 8 to 4 bits you can mostly get away with rounding (round-to-nearest, GPTQ, AWQ) because the weights are forgiving and you only quantize weights. Below 4 bits — and especially the moment you also quantize *activations* down to 4 bits (the W4A4 regime) — three things happen at once: outliers in the activations blow up your quantization grid, the rounding error per weight stops being negligible, and the kernels that made 4-bit fast no longer apply. The rest of this post is a tour of the techniques that were invented to survive each of those failures: rotations, codebooks, native low-bit training, and the new floating-point-4 formats baked into Blackwell silicon.

Before we descend, here is the gap between the comfortable mental model and the reality below 4 bits. If you carry one table out of this article, carry this one.

| Assumption from the 4-bit era | The naive extrapolation | The reality below 4 bits |
| --- | --- | --- |
| "Lower bits is just a smaller scale factor" | Halve the bits, double the savings, lose a little accuracy linearly | Error grows *super-linearly*; 2-bit RTN is often unusable while 2-bit AQLM is near-lossless |
| "Quantize the weights, leave activations in FP16" | Activations are cheap, ignore them | At W4A4 the activation outliers, not the weights, are what destroy you |
| "Calibration data barely matters" | A few hundred random tokens is fine | Below 4 bits, calibration domain and even sequence length visibly move eval scores |
| "Fewer bits means faster" | 4-bit is 4x less memory so it is ~4x faster | Only when memory-bound; at high batch you are compute-bound and dequant overhead can make it *slower* |
| "It is a post-training step" | Always quantize an existing FP checkpoint | The best sub-2-bit results come from *training* in low precision (BitNet), not quantizing afterward |
| "INT is the only integer-ish option" | INT4/INT8 everywhere | FP4 (MXFP4/NVFP4) with per-block scales now beats INT4 on Blackwell, in hardware |

> The single most expensive misconception in this space is treating "go lower" as a scalar knob. It is not a knob. It is a cliff with a different ladder bolted to the far side of each ledge.

Let me build the picture one failure at a time, starting with the wall that everyone hits first.

## 1. The outlier wall: why W4A4 breaks

**Senior rule of thumb: weight-only quantization is an accuracy problem; weight-and-activation quantization is an outlier problem. They are not the same problem and they do not have the same solution.**

When you keep activations in FP16 and only quantize weights, the weights of a trained transformer are remarkably well-behaved — roughly Gaussian, no catastrophic outliers, and a per-group scale captures them with a couple of bits of headroom. That is why GPTQ and AWQ work so well at 4-bit weight-only (W4A16). The trouble starts when you try to also quantize activations to low bit-width, which you must do if you want to use the cheap low-precision tensor-core math paths rather than just saving memory. The activations of a large language model are *not* well-behaved.

![Two giant channels force a wide per-tensor scale, collapsing the rest of the signal](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-2.webp)

The figure shows the problem in its starkest form. In the residual stream of a large transformer, a small number of *channels* — often fewer than ten out of thousands — carry activation magnitudes 50 to 100 times larger than the rest. These "massive activations" or "outlier features" were first cataloged carefully in the LLM.int8() work and have been confirmed in essentially every model family since. They are not noise; they are load-bearing. Zeroing them out wrecks the model. But they are poison for a quantizer.

Here is why. A symmetric integer quantizer with bit-width $b$ represents a tensor by a single scale $s = \max(|x|) / (2^{b-1} - 1)$ and stores each value as $q = \text{round}(x / s)$. The reconstruction is $\hat{x} = s \cdot q$, and the worst-case error per element is $s/2$. The scale $s$ is set by the *largest* magnitude in the group. So if one channel has magnitude 70 and the rest have magnitude around 1, then with 4-bit ($2^{3} - 1 = 7$ levels per sign) the scale is $s = 70/7 = 10$. Every "normal" value near 1 now rounds to either 0 or $\pm 10$ — the entire useful signal collapses onto two or three of the sixteen available levels. You have spent fifteen of your sixteen codes representing the empty space between the outlier and everything else.

This is the **outlier wall**. At 8-bit it is survivable: 256 levels give enough resolution that even a wide scale leaves a few bits for the normal channels. At 4-bit it is fatal. The classic round-to-nearest (RTN) W4A4 quantization of a Llama-class model loses 5 to 50 points of accuracy depending on the task, and on generative tasks the model often degenerates into repetition or gibberish. The weights were fine. The activations killed you.

It helps to be precise about *which* outliers you are fighting, because the fix depends on it. There are three distinct outlier phenomena, and conflating them is how people apply the wrong tool:

| Outlier type | Where it lives | What it breaks | The right fix |
| --- | --- | --- | --- |
| Per-channel activation outliers | A few hidden-dim channels, large across all tokens | Per-tensor and per-token activation scales | Rotation (spreads energy) or per-channel handling |
| Per-token activation spikes | Specific positions (often the BOS token) | Per-tensor activation scale | Per-token scaling; keep BOS in higher precision |
| Weight outliers | Scattered large weights | Weight group scale at very low bit-width | Group-wise scaling; rotation for incoherence |

The first row is the killer for W4A4 and the reason rotations exist. Per-token spikes are real but cheaper to handle — a per-token activation scale absorbs them, and many implementations simply keep the begin-of-sequence token's activations in higher precision because it is a notorious offender. Weight outliers are the mildest of the three and the reason group-wise weight scaling (a scale per 128 weights) was enough for the 4-bit weight-only era. When you read a method's claims, the first question to ask is which of these three it actually addresses: a weight-only method says nothing about the activation outliers that dominate W4A4.

There are three families of escape, and the rest of the article is largely about the first two:

1. **Make the outliers disappear before quantizing.** This is the rotation idea (QuaRot, SpinQuant): apply an orthogonal transform that spreads the outlier energy across all channels so no single one dominates the scale. This is the cleanest fix for W4A4 and the one I reach for first.
2. **Spend your bits more cleverly.** If you cannot get the values onto a uniform grid cheaply, stop using a uniform grid. Vector quantization (QuIP#, AQLM, VPTQ) maps *groups* of weights to entries in a learned codebook, which packs the representable points where the weights actually live. This is the path to genuinely good 2-bit and below.
3. **Keep a few outliers in high precision.** Mixed-precision schemes (the descendants of LLM.int8()) detect the outlier channels and run them in FP16 while the rest go low-bit. It works but it fragments the kernel and is increasingly the fallback rather than the frontier; rotations made it largely unnecessary for weights.

A note on terminology that trips people up in interviews: **W4A4** means 4-bit weights and 4-bit activations; **W4A16** means 4-bit weights with 16-bit activations (the comfortable weight-only regime); **W4A4KV4** adds a 4-bit KV cache. The harder the activation quantization, the more you need the techniques below. Weight-only 4-bit barely needs any of them.

## 2. Rotation: making a layer quantization-friendly

**Senior rule of thumb: if you can multiply a tensor by an orthogonal matrix for free, you can almost always make it easier to quantize without changing what the network computes.**

The single most important idea in modern low-bit quantization is also one of the most elegant, and it rests on a fact you already know from linear algebra: an orthogonal matrix $Q$ satisfies $Q Q^\top = I$, so you can insert $Q^\top Q$ anywhere in a chain of matrix multiplies without changing the result. If a linear layer computes $y = xW$, then for any orthogonal $Q$,

$$ y = xW = (xQ)(Q^\top W). $$

The output is mathematically identical. But the *distributions* of $xQ$ and $Q^\top W$ — the things the quantizer actually sees — can be enormously friendlier than $x$ and $W$. This is called **computational invariance**, and it is the foundation of QuaRot, SpinQuant, QuIP#, and most of what follows.

![Wrapping the matmul in a Hadamard rotation and its inverse erases activation outliers without changing the output](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-3.webp)

The figure captures the whole trick. On the left, the raw activations $x$ have two enormous outlier channels. Multiply by a **Hadamard matrix** $H$ — a special orthogonal matrix whose entries are all $\pm 1/\sqrt{d}$ — and the outlier energy is spread across every channel, producing a distribution that is close to Gaussian with no single dominant value. That rotated tensor quantizes beautifully into a uniform 4-bit grid because the scale is now set by a value only modestly larger than the typical one. And because $H H^\top = I$, the layer's output is unchanged: the bottom banner in the figure is the invariance identity, $y = xW = (xH)(H^\top W)$.

Why a Hadamard matrix specifically, and not a random rotation? Two reasons. First, the **central limit effect**: each output channel of $xH$ is a sum of $d$ terms with random signs, so by the central limit theorem it concentrates toward a Gaussian regardless of how spiky $x$ was. A single outlier of magnitude 70 contributes $70/\sqrt{d}$ to each channel — for $d = 4096$ that is about 1.1, completely tamed. Second, the **fast Walsh–Hadamard transform** computes $xH$ in $O(d \log d)$ time without ever materializing $H$, so the rotation that must happen at runtime is cheap. This is the property that the quantization literature calls *incoherence*: a Hadamard rotation makes the weight matrix incoherent, meaning its energy is evenly spread and no coordinate is special.

Here is the idea in twenty lines of NumPy. It demonstrates both halves: that the rotation is invariant, and that it crushes the dynamic range the quantizer has to cope with.

```python
import numpy as np
from scipy.linalg import hadamard

d = 4096                                   # hidden size (power of two for a dense Hadamard)
H = hadamard(d).astype(np.float32) / np.sqrt(d)   # normalized so H @ H.T == I

rng = np.random.default_rng(0)
x = rng.standard_normal((1, d)).astype(np.float32)
x[0, 137] =  60.0                          # inject a massive outlier channel
x[0, 891] = -45.0                          # and a second one (typical of LLM activations)
W = rng.standard_normal((d, d)).astype(np.float32) * 0.02

## 1. Computational invariance: rotating x by H and W by H.T leaves the output identical.
y_ref = x @ W
y_rot = (x @ H) @ (H.T @ W)
print("max |y_ref - y_rot|:", np.abs(y_ref - y_rot).max())     # ~1e-4, just fp32 noise

## 2. What the rotation does to the range the quantizer must cover.
rng_of = lambda v: float(v.max() - v.min())
print("dynamic range x      :", round(rng_of(x[0]), 1))        # ~105 -> outlier-dominated
print("dynamic range x @ H  :", round(rng_of((x @ H)[0]), 1))  # ~8   -> outliers spread out
```

The output range drops from about 105 to about 8 — more than a 10x reduction in the span the 4-bit grid must cover, which directly buys you roughly 3.5 extra effective bits of resolution on the normal channels. That is the difference between W4A4 being unusable and W4A4 being within a couple of points of FP16.

### How QuaRot uses this in practice

[QuaRot](https://arxiv.org/abs/2404.00456) (Ashkboos et al., 2024) was the method that turned this from a theoretical nicety into a deployable W4A4 recipe. It applies rotations at carefully chosen points so that *every* matmul in the transformer sees rotated, outlier-free inputs, and it does so with zero runtime overhead for most of them by *fusing* the rotation into the adjacent weight matrices offline. QuaRot demonstrated 4-bit quantization of weights, activations, and KV cache (W4A4KV4) on models up to Llama-2-70B while retaining more than 99% of the full-precision accuracy on several language understanding tasks — a result that was, frankly, shocking when it landed, because W4A4 had been considered a graveyard.

A practical wrinkle worth knowing: a dense Hadamard matrix only exists for sizes that are powers of two (and a handful of special sizes). Real hidden dimensions do not always cooperate, but you can compose a small dense Hadamard with a random orthogonal matrix on the leftover factor, or use a *randomized* Hadamard — a Hadamard times a random sign diagonal — to get the incoherence property at any size. The random sign diagonal matters for a subtler reason too: a fixed Hadamard can occasionally align badly with a particular weight matrix, and randomizing the signs guarantees the incoherence property holds with high probability rather than relying on luck. This is why the papers say "randomized Hadamard transform" rather than just "Hadamard" — the randomization is doing real work, not decoration.

### Second-order optimization: the cost of the online rotations

The catch is that not every rotation can be fused offline. A rotation that sits between two weight matrices can be folded into them once, for free. But a rotation that sits *inside* the attention computation — between the value projection and the output projection, where there is a non-linear softmax-weighted sum in the middle — has to be applied at runtime, per token, on live activations. These are the "online Hadamards," and they cost real FLOPs (cheap, thanks to the fast transform, but not zero) and real kernel complexity. The art of a good rotation scheme is to fuse as many rotations as possible and minimize the online ones. That placement question is its own topic, which is the next section.

## 3. Learned rotations and where they live

**Senior rule of thumb: a random Hadamard is a great default; a rotation learned on your calibration data is several accuracy points better, and at W4A4 those points are the difference between shippable and not.**

QuaRot uses *fixed* random Hadamard matrices. [SpinQuant](https://arxiv.org/abs/2405.16406) (Liu et al., 2024) asked the obvious follow-up question: if a rotation is free to insert, why not *learn* the best one for this specific model? The rotation matrix is constrained to be orthogonal, which means it lives on the Stiefel manifold (the set of matrices with orthonormal columns), and you can optimize over that manifold with Cayley-transform SGD — a gradient method that takes steps while staying on the manifold so the matrix remains orthogonal at every iteration. You minimize the end-to-end quantization error of the network with respect to the rotation, on a small calibration set, for a few hundred steps.

The payoff is consistent and meaningful. On zero-shot reasoning tasks at W4A4KV4, SpinQuant narrows the gap to full precision to about 2.9 points on Llama-2-7B. On Llama-3-8B it cuts the gap to roughly 4.4 points, versus about 6.3 points for QuaRot's fixed rotation; on Llama-3-70B the improvement over QuaRot is even larger. Learning the rotation does not change the asymptotics — you still pay for online Hadamards at runtime — but it is a one-time offline cost that meaningfully shifts the accuracy frontier. For a model you are going to serve millions of times, spending an afternoon learning the rotation is among the highest-leverage things you can do.

### Where the rotations actually go

The reason rotation methods feel fiddly to implement is that a transformer block is not one matmul; it is a sequence of them with normalization, residual adds, and attention in between, and each junction has a different rotation story. The figure below traces a single block and tags each linear with how its rotation is handled.

![Most rotations fuse into neighbouring weights for free; only the ones inside attention run online per token](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-4.webp)

Walking the block left to right: the hidden state enters, gets RMS-normalized, and flows into the QKV projection. The rotation here can be **fused offline** into the QKV weight (and a matching rotation absorbed into the previous block's output projection and the embedding), so it costs nothing at runtime. Inside attention, after the value projection and before the output projection, there is a rotation that must run **online** — it sits across the softmax-weighted value aggregation and cannot be folded into a static weight. The output projection's rotation fuses again. Then RMSNorm, the MLP up-projection (fused), and finally the MLP down-projection, whose input rotation is the second mandatory **online** Hadamard because it sits after the non-linear gating. So in a typical scheme you fuse four rotations into weights for free and pay for two online Hadamards per block. The online ones are the engineering cost; everything else is a one-time offline rewrite of the checkpoint.

How expensive are those two online Hadamards per block, really? Each is a fast Walsh-Hadamard transform of the activation, $O(d \log d)$ for hidden size $d$ — for $d = 4096$ that is about twelve cheap passes over the activation, small relative to the matmul it precedes. The cost that actually bites is not FLOPs but kernel integration: the online Hadamard has to be fused into the surrounding quantize-matmul-dequantize pipeline, or it becomes a separate kernel launch that reads and writes the activation to HBM, and that memory round-trip can cost more than the transform's arithmetic. Mature W4A4 implementations fuse the Hadamard into the quantization kernel so the activation is rotated in registers or shared memory on its way into the low-bit matmul. If your rotated model is accurate but no faster than FP16, an unfused online Hadamard is the first place to look.

### Second-order optimization: rotations interact with normalization

A subtle gotcha that bites people: rotations and RMSNorm do not commute trivially. RMSNorm scales by a per-token norm and then multiplies by a learned diagonal gain $\gamma$. To fuse a rotation cleanly through a normalization layer, you have to absorb the $\gamma$ into the following linear first (turning RMSNorm into a plain normalization), then the rotation passes through the now-pure normalization because scaling by a scalar commutes with an orthogonal rotation. Skipping this $\gamma$-absorption step is the most common reason a hand-rolled QuaRot implementation produces a model that is subtly worse than the paper — the rotation is no longer exactly invariant, and you have injected a small systematic error into every layer. If your rotated model is mysteriously a point or two below the reported numbers, check the normalization fusion first.

## 4. Codebooks: vector quantization for the 2-bit regime

**Senior rule of thumb: below about 3 bits, stop rounding individual weights to a uniform grid and start mapping groups of weights to a learned codebook. A uniform grid wastes its precision on values that never occur.**

Rotation makes a uniform grid *work* at 4 bits by reshaping the distribution to fit the grid. But at 2 bits a uniform grid has only four levels per sign, and no amount of reshaping makes four levels enough to represent a weight to useful precision. The problem is the grid itself: it is one-dimensional and uniform, so it spends representable points on the entire real line whether or not weights ever land there. The fix is to change the geometry of the representable set.

![Scalar quantization snaps each weight independently; vector quantization snaps weight groups to dense lattice points](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-5.webp)

The figure contrasts the two geometries. On the left, **scalar quantization**: each weight is rounded to the nearest point on an axis-aligned grid, independently of its neighbors. A weight that falls between grid lines (the red point) is forced to the nearest level, and at low bit-width those levels are far apart, so the rounding error is large. On the right, **vector quantization**: you take a *group* of weights — a pair, or a block of eight — and treat it as a point in a higher-dimensional space, then snap that point to the nearest entry in a learned codebook (the green points). Because the codebook is learned from the actual weight distribution and can place its points anywhere in the space (a dense lattice, not an axis-aligned grid), it covers the region where weights actually live far more efficiently. The same number of stored bits buys you a much smaller reconstruction error.

The bit accounting is the key. If you encode groups of $g$ weights with an index into a codebook of size $2^k$, you spend $k$ bits per group, or $k/g$ bits per weight. With $g = 8$ and an 8-bit index ($k = 8$) you are at exactly 1 bit per weight from a single codebook; stack a few codebooks and add their entries and you climb to 2 or 3 bits per weight with dramatically better fidelity than scalar quantization at the same budget.

### The two leading codebook methods

[QuIP#](https://arxiv.org/abs/2402.04396) (Tseng et al., 2024) combines two ideas: it first applies a Hadamard rotation for incoherence (the same trick from Section 2, here used to make the *weights* well-conditioned for coding), then quantizes the rotated weights with a codebook based on the **E8 lattice** — the densest known lattice packing in eight dimensions. The E8 lattice is the mathematically optimal way to place points in 8-D space so that any random point is close to one of them, which is exactly what you want for minimizing quantization error. QuIP# at 2 bits achieves a WikiText-2 perplexity around 4.16 on Llama-2-70B, close to the FP16 baseline and far ahead of any scalar 2-bit method.

[AQLM](https://arxiv.org/abs/2401.06118) (Egiazarian et al., 2024) — Additive Quantization of Language Models — takes the additive-codebook route shown in the next figure. Instead of one big codebook, it uses several small ones and represents each weight group as the *sum* of one entry from each.

![Summing a few entries from small learned codebooks represents a weight group far more cheaply than one giant table](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-6.webp)

The figure traces the reconstruction. A weight block is encoded into a handful of 8-bit indices (one per codebook), each index selects a vector from its codebook, and the selected vectors are summed to approximate the original block. The reason this is cheap is combinatorial: $M$ codebooks of size 256 each give you $256^M$ representable combinations while storing only $8M$ bits and keeping the codebooks tiny enough to live in cache. AQLM was the first method shown to be Pareto-optimal below 3 bits per parameter, reaching a WikiText-2 perplexity of about 3.94 on Llama-2-70B at roughly 2 bits — better than QuIP# in that early comparison, though the two methods have leapfrogged each other since and both are excellent. VPTQ (Vector Post-Training Quantization) is a third strong entry in the same family, using second-order information to design the codebooks.

To put the codebook methods on one axis with the scalar baselines, here is the rough accuracy picture at the extreme-compression end, on Llama-2-70B at roughly 2 bits per weight (WikiText-2 perplexity, lower is better; the FP16 baseline is about 3.3):

| Method | Bits/weight | Wiki2 PPL (Llama-2-70B) | Decode speed | Notes |
| --- | --- | --- | --- | --- |
| RTN scalar | 2 | unusable (>100) | fast | uniform grid collapses |
| GPTQ scalar | 2 | very poor | fast | scalar grid, no codebook |
| QuIP# | ~2 | ~4.16 | medium | Hadamard incoherence + E8 lattice |
| AQLM | ~2 | ~3.94 | medium | additive codebooks, matching pursuit |
| VPTQ | ~2 | competitive | medium | second-order codebook design |

The gap between the scalar rows and the codebook rows is the entire argument for vector quantization: at 2 bits, scalar methods are unusable and codebook methods land within a fraction of a perplexity point of FP16. AQLM and QuIP# have traded the lead back and forth across releases; treat them as equally strong and choose on toolchain support and kernel availability for your stack rather than on a stale perplexity comparison.

Here is the additive-codebook idea as runnable NumPy. The `encode` function is a stripped-down residual matching pursuit — the cheap inner loop at the heart of AQLM — and `decode` is what the GPU does at inference time.

```python
import numpy as np

rng = np.random.default_rng(0)
d, g = 4096, 8                # number of weight groups; vector length per code
n_codebooks = 2              # AQLM-style: weight block ~= sum of M codes
codebook_size = 256          # 8-bit index per codebook

## Learned codebooks: each maps an 8-bit index -> a length-g vector.
codebooks = [rng.standard_normal((codebook_size, g)).astype(np.float32) * 0.02
             for _ in range(n_codebooks)]

## A weight matrix viewed as d groups, each a length-g vector.
W = rng.standard_normal((d, g)).astype(np.float32) * 0.02

def encode(w, codebooks):
    """Greedy residual matching pursuit: subtract the best code, repeat."""
    residual, idx = w.copy(), []
    for C in codebooks:
        dists = ((C[None, :, :] - residual[:, None, :]) ** 2).sum(-1)  # (d, 256)
        j = dists.argmin(1)                                            # (d,)
        idx.append(j)
        residual = residual - C[j]
    return np.stack(idx, 1)                                           # (d, M)

def decode(idx, codebooks):
    return sum(C[idx[:, m]] for m, C in enumerate(codebooks))

idx   = encode(W, codebooks)
W_hat = decode(idx, codebooks)

bits_per_weight = n_codebooks * np.log2(codebook_size) / g
print(f"bits / weight : {bits_per_weight:.2f}")                       # 2.00 bits
print("relative error:", float(np.linalg.norm(W - W_hat) / np.linalg.norm(W)))
```

### Second-order optimization: codebook quantization is decode-bound

The accuracy is wonderful; the speed is the catch, and it is the opposite of the scalar story. Scalar 4-bit quantization is fast because dequantization is a multiply by a scale — one cheap op per weight. Codebook decoding is a *gather*: for every weight group you index into one or more codebook tables and sum the results. Gathers are memory-indirection-heavy and do not map cleanly onto tensor cores. Early AQLM and QuIP# inference was substantially slower than FP16 for small batches despite using a quarter of the memory, because the decode step dominated. Both projects invested heavily in fused decode kernels, and the situation is much better now, but the lesson stands: **vector quantization trades compute and kernel complexity for storage**. Use it when memory is the hard constraint (fitting a 70B on a 24 GB card) and you can tolerate the decode cost, not when you are chasing raw throughput on hardware that already fits the model.

## 5. Native low-bit: BitNet and ternary weights

**Senior rule of thumb: the best way to get a great 1.58-bit model is not to quantize a 16-bit model. It is to train a 1.58-bit model from scratch. Post-training quantization is recovery; native training is prevention.**

Everything so far has been *post-training quantization* (PTQ): take a finished FP checkpoint and compress it, fighting to recover the accuracy the compression destroyed. There is a fundamentally different path. What if the model never had high-precision weights to lose? What if it learned, during training, to be good *with* low-precision weights? That is **quantization-aware training (QAT)** taken to its logical extreme, and the landmark result is [BitNet b1.58](/blog/paper-reading/large-language-model/bitnet-b1-58-2b4t-technical-report).

BitNet b1.58 uses **ternary weights** — every weight is one of $\{-1, 0, +1\}$, which is $\log_2 3 \approx 1.58$ bits of information. The "b1.58" in the name is literally that entropy. The model is trained from scratch with these ternary weights in the forward pass, and the extraordinary claim, now well-supported, is that at the 2-billion-parameter, 4-trillion-token scale (the "2B4T" model), it matches the downstream accuracy of full-precision open models of the same size while using a fraction of the memory and energy.

![Native training keeps full-precision latent weights but uses ternary weights in the forward pass, turning matmuls into adds](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-7.webp)

The figure shows the mechanism, and there are two ideas in it that you have to get right. First, **there is still a full-precision weight** — the "latent weight" $W$ in FP32. You cannot do gradient descent on a discrete set of three values; gradients are infinitesimal and the ternary set has no useful gradient. So you keep a continuous shadow copy, and on each forward pass you ternarize it on the fly. The ternarization uses a per-tensor absmean scale: $\gamma = \text{mean}(|W|)$, and each weight becomes $\text{round}(W/\gamma)$ clamped to $\{-1, 0, +1\}$.

Second, the gradient has to get *back* to the latent weight through a rounding operation that has zero gradient almost everywhere. The trick is the **straight-through estimator (STE)**: in the backward pass you pretend the rounding was the identity function, so $\partial L / \partial W \approx \partial L / \partial \hat{W}$. The dashed arrow in the figure is the STE — the gradient flows straight through the ternarization back to the full-precision copy, which accumulates the tiny updates until a weight eventually flips from, say, $0$ to $+1$. The forward pass is ternary; the learning is continuous.

The reason this is worth the trouble is the forward pass itself: when weights are in $\{-1, 0, +1\}$, the matrix multiply $Y = XW$ has no multiplications. Each weight either adds the activation, subtracts it, or ignores it. On hardware that is bottlenecked on multiply-accumulate units, replacing multiplies with adds is a large energy and area win — reported energy figures for BitNet inference are around an order of magnitude below a comparable FP model. This is the property that makes native 1-bit-class models interesting for CPUs and edge accelerators, not just for memory savings.

Here is the BitLinear forward in PyTorch, including the STE done the idiomatic way with a detached correction term.

```python
import torch
import torch.nn as nn

def ternarize(w: torch.Tensor):
    # BitNet b1.58: per-tensor absmean scale, round to {-1, 0, +1}.
    gamma = w.abs().mean().clamp(min=1e-5)
    w_tern = torch.clamp(torch.round(w / gamma), -1, 1)
    return w_tern, gamma

class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        w_tern, gamma = ternarize(w)
        # Straight-through estimator: forward value is the ternary weight,
        # but the gradient flows to the full-precision `w` as if identity.
        w_ste = w + (w_tern * gamma - w).detach()
        return nn.functional.linear(x, w_ste, self.bias)   # 8-bit act-quant elided

layer = BitLinear(4096, 4096, bias=False)
y = layer(torch.randn(2, 4096))
y.sum().backward()
print("dense gradient via STE:", layer.weight.grad.abs().mean().item() > 0)   # True
```

### Quantization-aware fine-tuning: the middle path

Between full native low-bit pre-training and pure post-training quantization sits a pragmatic middle: take a finished FP checkpoint, apply your PTQ method, then run a short *quantization-aware fine-tune* to let the model adapt to the quantization it now lives with. This is far cheaper than pre-training — hundreds of millions of tokens, not trillions — and it recovers a meaningful slice of the accuracy that aggressive PTQ gave up. The straight-through estimator from the BitNet code is exactly the tool: you fine-tune with the quantizer in the forward pass and the STE carrying gradients to a full-precision shadow copy, so the model learns weights that are good *after* quantization rather than before.

The most common production form is low-rank QAT: freeze the quantized base, add small LoRA adapters in higher precision, and fine-tune only those. This is the QLoRA lineage, and it doubles as both a fine-tuning method and an accuracy-recovery method for the quantized base. The trade is real but modest — you need a fine-tuning dataset and a few GPU-hours — and the payoff is often a point or two of recovered accuracy at 2-3 bits, sometimes the difference between shippable and not. For the broader fine-tuning context, see [effective LLM fine-tuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques). When PTQ alone leaves you just short of your accuracy bar and you already have a fine-tuning pipeline, a short QAT pass is usually a better next step than switching quantization methods entirely.

### Second-order optimization: native low-bit is not a free lunch you can apply later

The obvious objection is cost. Native low-bit training is *training* — you pay the full pre-training bill, and you cannot retrofit it onto a model someone else trained in FP16. That is the real reason BitNet-style models are not everywhere despite their elegance: almost nobody has both the desire for a 1.58-bit model and the budget to pre-train one from scratch on trillions of tokens. The methods in the previous sections exist precisely because PTQ lets you compress *someone else's* finished model. BitNet is the right answer when you control training and ship at enormous scale (so the per-inference energy savings dominate the one-time training cost), or on a target where the multiply-free forward pass is the whole point. For most teams quantizing an off-the-shelf Llama or Qwen, native training is not on the table — but it is essential context for understanding why the PTQ methods have to work so hard: they are trying to recover, after the fact, what BitNet got for free by never throwing it away.

## 6. New silicon, new formats: MXFP4 and NVFP4

**Senior rule of thumb: INT4 was the right format when the hardware had no native 4-bit path. On Blackwell there is one, and it is floating point, not integer. Quantize to the format your tensor cores actually execute.**

Everything so far has been a software fight against hardware that natively supports, at best, INT8 and FP16. The ground shifted with NVIDIA Blackwell (the B100/B200 generation), which added native tensor-core support for 4-bit *floating-point* formats. This changes the calculus, because now the cheapest precision is not a software emulation — it is a hardware instruction running at up to 2x the FP8 rate and 4x the BF16 rate.

The two formats that matter are **MXFP4** and **NVFP4**, and both are built on the same idea: **microscaling**. A plain FP4 number — the E2M1 format, with 1 sign bit, 2 exponent bits, and 1 mantissa bit — can represent only the magnitudes $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$. That is hopeless as a standalone format; the dynamic range is tiny. Microscaling rescues it by giving every small *block* of values its own shared scale factor, so the FP4 values encode the fine structure within a block while the block scale handles the magnitude.

![FP4 only works because each small block carries its own scale; NVFP4's 16-wide blocks beat MXFP4's 32-wide blocks](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-8.webp)

The figure lays out the difference between the two formats, which is entirely in the granularity and type of that shared scale. **MXFP4** (the Open Compute Project microscaling standard) uses blocks of **32** elements, each with a shared scale stored as an **E8M0** value — a raw power-of-two exponent, 8 bits, no mantissa. **NVFP4** (NVIDIA's variant) uses blocks of **16** elements with a shared scale stored as an **E4M3 FP8** value — a real floating-point scale with a mantissa, not just a power of two. Both changes push in the same direction: smaller blocks mean the scale adapts to local structure more tightly, and a floating-point scale represents the per-block magnitude more precisely than a coarse power of two. The net result is that NVFP4 preserves model quality noticeably better than MXFP4 at the same 4-bit storage.

The empirical results are why this section exists. NVFP4 inference on Blackwell delivers roughly **2.3x higher throughput than INT4** at matched accuracy, and near-lossless quality versus FP16/BF16 on knowledge, math, and commonsense benchmarks. More strikingly, NVIDIA demonstrated NVFP4 used for *pre-training* at scale — a 12-billion-parameter model trained with NVFP4 tracked an FP8 baseline's loss curve closely, with the small remaining gap not translating into measurable downstream degradation, and roughly 3x faster time-to-train versus the previous Hopper generation. Four-bit floating point went from "a desperate inference compression" to "a viable training precision" in about one hardware generation.

Here is the microscaling decode, which is all the GPU does to turn a block of FP4 codes back into usable numbers.

```python
import numpy as np

## E2M1 lookup table: the 8 representable FP4 magnitudes, with sign in the high bit.
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)

def decode_mx_block(codes_4bit, scale_exp_e8m0):
    """MXFP4: 32 FP4 elements sharing one E8M0 (power-of-two) block scale."""
    scale = np.float32(2.0) ** np.float32(int(scale_exp_e8m0) - 127)   # E8M0 bias = 127
    return E2M1[codes_4bit] * scale

block = np.array([2, 3, 6, 1, 0, 9, 12, 7] + [0] * 24, np.uint8)       # 32 codes
print(decode_mx_block(block, scale_exp_e8m0=131)[:8])

## NVFP4 differs in exactly two ways the figure calls out:
##   (1) block of 16 instead of 32  -> finer-grained scaling
##   (2) E4M3 FP8 scale instead of E8M0 -> the scale itself has a mantissa
## Both reduce per-block quantization error, which is why NVFP4 > MXFP4 at 4 bits.
```

One detail makes FP4 *training* (as opposed to inference) work where it naively would not: **stochastic rounding**. When you round a weight update to FP4 with plain round-to-nearest, any update smaller than half a quantization step rounds to zero and is lost forever — and across millions of steps those lost updates are most of the learning signal. Stochastic rounding instead rounds up or down with probability proportional to proximity, so a small update has a small *chance* of nudging the value to the next FP4 level; in expectation the update is preserved even though any single step is lossy. NVFP4 pre-training leans on this, plus a two-level scaling scheme: a fine per-block FP8 scale (the E4M3 we saw) combined with a coarse per-tensor scale that keeps the whole tensor inside FP4's representable range. The combination is what let a 12B model train in NVFP4 while tracking an FP8 loss curve. The lesson generalizes: low-precision *training* needs stochastic rounding and careful multi-level scaling that low-precision *inference* can skip, because inference never accumulates millions of tiny updates.

### Second-order optimization: do not stack a software 4-bit method on top of a hardware 4-bit format

A trap I have watched teams fall into: they have a great QuaRot or AQLM pipeline tuned for INT4, they get a Blackwell box, and they try to run their INT4 software path on hardware that wants NVFP4. You end up emulating, losing the native speed, and getting the worst of both worlds. On Blackwell, the right move is to quantize *to* NVFP4 directly with a tool that produces the hardware-native layout (llm-compressor's NVFP4 scheme, TensorRT-LLM's FP4 path) and let the tensor cores execute it. The rotation and outlier tricks still matter — they help the FP4 quantizer too, and the best Blackwell pipelines apply a rotation before NVFP4 quantization — but the *storage format* should match the silicon. Pick the format your hardware executes natively, then apply the accuracy tricks on top of it, not a different format underneath it.

## 7. The kernel reality: fewer bits is not free speed

**Senior rule of thumb: quantization is a memory-bandwidth optimization first and a compute optimization second. If you are not memory-bound, low-bit weights may not speed you up at all — and can slow you down.**

This is the section that separates people who have deployed quantized models from people who have only read about them. The seductive but wrong mental model is "4-bit is 4x less data, therefore 4x faster." The truth is governed by the roofline.

![At low batch you ride the memory ceiling and 4-bit wins; past the ridge point you become compute-bound and the win evaporates](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-9.webp)

The figure is two grouped-bar comparisons of throughput. On the left, at **batch size 1** (the single-user, autoregressive decode case), the GPU is **memory-bound**: it spends almost all its time reading weights from HBM, and the math units sit mostly idle. Here, cutting the weights from 16 bits to 4 bits cuts the dominant cost — weight traffic — by roughly 4x, and you get close to a 4x speedup. This is the regime where weight-only quantization shines, and it is exactly the regime of latency-sensitive single-stream serving.

On the right, at **batch size 64** (high-throughput batched serving), the GPU is **compute-bound**: the weights are read once and reused across all 64 sequences in the batch, so weight traffic is amortized and the bottleneck moves to the matrix-multiply FLOPs. Now the bit-width of the *weights* barely matters for speed, because you are limited by the multiply-accumulate units, not by memory. Worse, weight-only 4-bit has to **dequantize** the weights back to FP16 before the tensor cores can multiply them, and that dequant work is pure overhead that did not exist in the FP16 path. At high batch, a naive weight-only-4-bit kernel can be *slower* than FP16. The "W4 / INT4" bar on the right is shaded as a caution for exactly this reason.

The ridge point between these regimes is the whole game, and it is why kernel engineering is inseparable from quantization. The two kernels that define the state of the art on NVIDIA hardware are worth knowing by name:

- **Marlin** (from IST-DASLab) is a mixed-precision FP16-times-INT4 GEMM kernel engineered to keep the dequantization off the critical path. It pipelines the dequant into shared memory so the tensor cores never wait, and it sustains a near-ideal ~4x speedup up to medium batch sizes of 16 to 32 tokens — pushing the ridge point much further right than naive kernels. Marlin is the reason GPTQ/AWQ models in vLLM are fast at all: the same checkpoint that runs at 276 tok/s on a generic kernel runs at 700+ tok/s with Marlin.
- **Machete** (from the vLLM/Neural Magic side) is the Hopper-generation successor, rebuilt on the newer tensor-core instructions and the warp-group MMA path. With Machete, you can serve Llama-3.1-70B at 4-bit on a single H100 at roughly 5 requests per second while holding median time-to-first-token under 250 ms and time-per-output-token under 100 ms — the kind of number that makes single-GPU 70B serving economically real.

Here is what the production path actually looks like: produce a Marlin-compatible W4A16 checkpoint with llm-compressor, then serve it with vLLM, which auto-selects the right kernel.

```bash
## Produce a W4A16 GPTQ checkpoint with Marlin-compatible packing.
pip install llmcompressor vllm

python - <<'PY'
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

oneshot(
    model="meta-llama/Llama-3.1-70B-Instruct",
    dataset="open_platypus",                       # ~512 calibration sequences
    recipe=GPTQModifier(targets="Linear", scheme="W4A16",
                        group_size=128, ignore=["lm_head"]),
    output_dir="Llama-3.1-70B-W4A16-G128",
)
PY

## vLLM auto-selects Marlin/Machete for this packing on Ampere/Hopper.
vllm serve Llama-3.1-70B-W4A16-G128 \
  --quantization compressed-tensors \
  --max-model-len 8192 --gpu-memory-utilization 0.92
```

On Blackwell, the analogous recipe targets NVFP4 so the work lands on the native 4-bit tensor cores instead of an emulated INT4 path:

```bash
python - <<'PY'
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

oneshot(
    model="meta-llama/Llama-3.1-8B-Instruct",
    recipe=QuantizationModifier(targets="Linear", scheme="NVFP4",
                                ignore=["lm_head"]),
    output_dir="Llama-3.1-8B-NVFP4",
)
PY
vllm serve Llama-3.1-8B-NVFP4 --quantization compressed-tensors
```

There is a middle regime worth naming explicitly: **W4A8** — 4-bit weights with 8-bit activations. Pure weight-only W4A16 wins at low batch but adds dequant overhead at high batch; full W4A4 needs rotations and is fragile. W4A8 is the pragmatic compromise that a lot of production serving has converged on, because 8-bit activations are robust to outliers (256 levels absorb them without rotation) while 4-bit weights still cut the memory traffic that dominates at moderate batch. On hardware with fast INT8 or FP8 activation paths, W4A8 often gives the best throughput-accuracy point across a range of batch sizes, sidestepping both the high-batch dequant penalty of W4A16 and the outlier fragility of W4A4. If you are unsure where your batch profile sits, or it varies across the day, W4A8 — or plain FP8 weights-and-activations on Hopper and Blackwell — is the safe default that rarely embarrasses you. Reserve W4A4 for the cases where you have measured that the activation math path is your actual bottleneck and you are willing to carry the rotation machinery to make it accurate.

### Second-order optimization: match the method to the batch profile

The practical consequence of the roofline is that **the right quantization method depends on how you serve**. If you are a latency-sensitive single-user deployment (a local assistant, an on-device model, a low-QPS internal tool), you live at batch 1, you are memory-bound, and aggressive weight-only quantization is almost pure win. If you are a high-throughput batched API serving thousands of concurrent requests, you live at high batch, you are compute-bound, and the right lever is a format the tensor cores execute natively (FP8, or NVFP4 on Blackwell) rather than a weight-only INT4 scheme that adds dequant overhead. The worst outcome is choosing the method by "how many bits" instead of "what is my bottleneck." For the broader serving picture, the [optimizing LLM inference guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) and [vLLM internals](/blog/machine-learning/large-language-model/vllm-inference) go deeper on the throughput side; this section is just the quantization-shaped slice of it.

## 8. KV cache at the frontier

The weights are not the only thing worth quantizing. For long-context serving, the **KV cache** — the stored keys and values for every past token — grows linearly with sequence length and batch, and at long context it dominates memory, often exceeding the weights themselves. Quantizing it is a separate axis from quantizing the weights, with its own outlier story, and it is increasingly where the practical wins are. The general mechanics of KV cache memory are covered in [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management); here I want only the frontier-specific wrinkle.

To see why this matters, do the arithmetic. For a Llama-3-8B-class model the KV cache costs roughly 128 KB per token in FP16 (keys and values, all layers, all heads). At 8K context and batch 32 that is about 32 GB of KV cache — larger than the 16 GB of FP16 weights, and far larger than the weights once *those* are 4-bit. At long context the cache, not the weights, determines how many concurrent requests fit on a card. A 4-bit KV cache cuts that 32 GB to 8 GB; a 2-bit cache to 4 GB. That is the difference between serving 32 and 128 concurrent long-context requests on the same GPU, which is why KV quantization has become as important as weight quantization for any long-context workload.

The wrinkle is that **keys and values quantize differently**, because their outliers live in different places. Key tensors have strong *per-channel* outliers — certain channels are consistently large across all tokens — which means you should quantize keys per-channel (a scale per channel), because a per-token scale would be dominated by those outlier channels exactly as in Section 1. Value tensors are better behaved per-token. KVQuant and similar methods exploit precisely this asymmetry: per-channel quantization for keys, per-token for values, plus a small set of full-precision outliers. And — predictably, given everything above — applying a Hadamard rotation to the cache before quantizing helps here too, for the same incoherence reason it helps activations.

The numbers make the case: with the right per-channel handling, the KV cache can be pushed to 4-bit, and in some schemes to 2-bit, with negligible accuracy loss, which roughly doubles to quadruples the context length you can hold in a given memory budget. The failure mode if you get it wrong is specific and recognizable: quantize keys per-token and your long-context recall collapses — the model loses the ability to retrieve facts from early in a long prompt — while short prompts look fine, so it sails through a naive eval and fails in production. That exact incident is one of the case studies below.

A second frontier wrinkle worth flagging: KV-cache quantization interacts with **speculative decoding**. If you run a quantized draft model to propose tokens and a higher-precision target to verify, the draft's quantization affects the acceptance rate, not the final quality — so you can be far more aggressive quantizing the draft than the target. That decoupling is one of the cleaner free lunches at the frontier and pairs naturally with the techniques in this article.

## 9. Calibration, evaluation, and accuracy regressions

Every PTQ method in this article — GPTQ, AWQ, QuaRot, SpinQuant, QuIP#, AQLM, NVFP4 quantization — depends on a **calibration set**: a small sample of data used to estimate activation statistics, fit scales, learn rotations, or design codebooks. At 4-bit weight-only, calibration is forgiving; a few hundred random web-text sequences work fine. Below 4 bits, and at W4A4, calibration becomes a first-class accuracy lever, and getting it wrong is the most common cause of "the paper got X and I got X minus four points."

Three calibration rules that matter more the lower you go. First, **domain match**: the calibration data should resemble the deployment distribution. A model quantized on generic web text and deployed on code or math can lose several points that a domain-matched calibration set would have recovered, because the activation statistics — and therefore the scales and rotations fit to them — differ by domain. Second, **sequence length**: calibrate at a length representative of production, because activation outlier magnitudes grow with context and a model calibrated on 512-token sequences can misbehave at 8K. Third, **enough samples but not too many**: 128 to 512 sequences is the usual sweet spot; too few and the statistics are noisy, too many and you mostly waste time without improving accuracy.

The evaluation side is just as treacherous, and the deeper you quantize the more it matters. **Perplexity is necessary but wildly insufficient.** A 2-bit model can have a WikiText-2 perplexity within a hair of FP16 and still be visibly worse at multi-step reasoning, instruction following, or long-context retrieval, because perplexity averages over all tokens and quantization damage concentrates on the hard, information-dense tokens that perplexity barely weights. Always pair perplexity with task evals — GSM8K or MATH for reasoning, IFEval for instruction following, a long-context retrieval probe like needle-in-a-haystack — and run them at the context lengths and on the domains you actually serve. For the broader argument that perplexity is the wrong north star, see [evaluating conversational LLMs beyond perplexity](/blog/machine-learning/large-language-model/evaluating-conversational-llms-beyond-perplexity).

It helps to know which eval catches which quantization failure. Math word problems (GSM8K, MATH) are the most sensitive to weight-quantization damage because they require precise multi-step computation where one wrong token derails the whole chain — they often regress before perplexity moves at all. Instruction-following evals (IFEval) catch the subtler degradation where a model still produces fluent text but stops respecting constraints, a common low-bit failure. Long-context retrieval probes (needle-in-a-haystack at your real context length) are the only thing that catches KV-cache quantization damage, which is invisible to every short-prompt metric. And a small set of golden generations diffed by a judge model catches the qualitative "it just feels dumber" regressions that no benchmark names. Run all four; each is blind to what the others see, and shipping aggressive quantization on the strength of any single one is how a specific, recognizable failure reaches users.

The discipline that catches regressions before users do is an **accuracy regression suite** run on every quantized build: a fixed battery of task evals with hard thresholds, plus a small set of golden generations diffed by a judge model. Treat a 4-bit checkpoint exactly like a code change — it can introduce a subtle bug (a collapsed long-context recall, a math regression) that no single aggregate metric surfaces. The teams that ship aggressive quantization safely are the ones that test it like software, not the ones with the cleverest quantizer.

## Case studies from production

These are composite incidents drawn from the kinds of failures and fixes that recur across teams deploying low-bit models. Names and exact numbers are illustrative, but every failure mode is real and every root cause is one I have seen or could reproduce.

### 1. The W4A4 cliff that rotation rescued

A team had a clean W4A16 Llama-2-7B in production and decided to push to W4A4 to use the INT4 activation path on their hardware and double throughput. The naive round-to-nearest W4A4 build had a WikiText-2 perplexity that looked "only" a couple of points worse — but generation quality fell off a cliff: the model repeated phrases, lost coherence after a few sentences, and scored near zero on a reasoning eval. The wrong first hypothesis was "the weights need a better quantizer," and they spent a week tuning GPTQ damping with no effect. The actual root cause was the *activations*: a handful of outlier channels were blowing up the per-tensor activation scale, exactly the outlier wall. The fix was QuaRot — insert Hadamard rotations so the activations are outlier-free before the INT4 quantization — which recovered the reasoning eval to within about three points of FP16 with no change to the weight quantizer at all. The lesson: at W4A4, if generation is broken but perplexity looks only mildly worse, suspect activation outliers before you touch the weight pipeline.

### 2. SpinQuant's learned rotation closing the last gap

The same team, now on Llama-3-8B, found that QuaRot's fixed Hadamard left them about six points short of FP16 on their reasoning suite — close, but their product bar was "within four points." Rather than abandon W4A4, they switched the fixed rotation for a SpinQuant learned rotation: a few hundred steps of Cayley-SGD on the Stiefel manifold against 256 calibration sequences from their own domain. The learned rotation closed the gap to roughly four points, clearing the bar, at the cost of one afternoon of offline optimization and zero additional runtime cost (the learned rotation fuses and runs exactly like the fixed one). The lesson: when a fixed rotation gets you close, learning the rotation is the cheapest remaining point of accuracy you can buy, and it is a one-time offline cost amortized over every future inference.

### 3. A 70B on a 24 GB card via 2-bit codebooks

A researcher wanted to run Llama-2-70B on a single 24 GB consumer GPU for offline experimentation. At 4-bit weight-only the model is ~35 GB and does not fit. AQLM at roughly 2 bits brought the weights to ~18 GB, leaving room for a modest KV cache, and the WikiText-2 perplexity landed under 4 — eminently usable for the research workload. The surprise came at runtime: decode was meaningfully slower than the FP16 baseline on the few layers that fit, because the additive-codebook gather did not map onto tensor cores and the early fused decode kernels were immature. For a batch-1 offline workload this was an acceptable trade — slow but *possible* beats fast but *impossible* — but it would have been the wrong choice for a latency-sensitive product. The lesson: codebook quantization buys you memory you cannot otherwise have, at a decode-speed cost that is fine for offline and often wrong for serving.

### 4. BitNet matching a full-precision model at a fraction of the energy

A hardware-adjacent team building an on-device assistant could not fit a competitive FP model in their power budget — the multiply-accumulate energy of an 8-bit model exceeded their thermal envelope. Rather than quantize harder, they adopted a native BitNet b1.58 2B4T model, whose ternary weights replace the dominant matmul multiplications with additions. On their math and commonsense evals the ternary model was competitive with full-precision open models in the 2B class, while measured inference energy was roughly an order of magnitude lower, because the forward pass is multiply-free. The catch they accepted going in: this only worked because a strong native-trained ternary checkpoint already existed for their scale; they had no budget to train one themselves. The lesson: native low-bit is a different economic model — you either inherit a pre-trained native checkpoint or you have the scale to justify training one, and when either holds it dominates PTQ on energy.

### 5. "GPTQ was slower than FP16" — the missing Marlin kernel

A team reported that their freshly quantized GPTQ Llama model was *slower* than the FP16 original — 276 tok/s versus the FP16 baseline — and concluded that quantization "doesn't actually help for throughput." The wrong hypothesis was that the model was compute-bound. The actual root cause was the kernel: their serving stack was falling back to a generic, unfused INT4 GEMM that dequantized on the critical path, so every weight was unpacked to FP16 with the tensor cores stalled waiting. Enabling the Marlin kernel — which pipelines dequant into shared memory — took the same checkpoint to 700+ tok/s, a ~2.6x jump, with no change to the weights. The lesson: a quantized model is only as fast as its kernel, and "quantization didn't help" almost always means "I am not running the fused kernel." Always confirm which GEMM kernel is actually executing before drawing conclusions about quantization speed.

### 6. NVFP4 versus MXFP4 on a new Blackwell cluster

A team migrating to Blackwell quantized a model to MXFP4 because it was the open standard they had read about, and found the accuracy gap to FP8 larger than they wanted on a knowledge-heavy eval. Switching the same pipeline to NVFP4 closed most of the gap, for a reason that is entirely in the format: NVFP4's 16-element blocks (versus MXFP4's 32) and FP8 E4M3 block scale (versus MXFP4's raw power-of-two E8M0) give finer, more precise per-block scaling, so the same 4 bits of value storage carry less quantization error. Throughput was also better, since NVFP4 is the format NVIDIA's kernels are most tuned for, landing around 2.3x over their old INT4 path at matched accuracy. The lesson: on Blackwell the format choice within FP4 is not cosmetic — finer block scaling measurably wins, and the hardware-native format is also the fast one.

### 7. Machete making single-GPU 70B serving economical

An inference team needed to serve Llama-3.1-70B with interactive latency but had only single H100 nodes, not the multi-GPU setups a 70B usually demands. Weight-only 4-bit brought the model under 40 GB so it fit one card, but the open question was whether a single card could hit their latency SLOs under real concurrency. Using the Machete mixed-input kernel on Hopper, they sustained roughly 5 requests per second with median time-to-first-token under 250 ms and time-per-output-token under 100 ms — interactive, on one GPU, for a 70B model. The economics flipped: a workload that would have required a multi-GPU box per replica now ran one replica per H100. The lesson: the combination of 4-bit weights and a Hopper-class fused kernel is what makes single-GPU large-model serving viable; neither the quantization nor the kernel alone gets you there.

### 8. The KV-key outlier that broke long-context recall

A team quantizing the KV cache to 4-bit to extend their serveable context length shipped a build that passed all their short-prompt evals and then started failing in production on long documents — users reported the model "forgetting" facts stated early in long prompts. Short prompts were fine; long ones degraded. The wrong first hypothesis was a bug in the long-context attention path. The actual root cause was the KV quantization granularity: they had quantized *keys* per-token, and key tensors have strong per-channel outliers, so the per-token scale was dominated by a few outlier channels and the rest of each key vector was crushed — exactly the outlier wall, now in the cache. The fix was to quantize keys *per-channel* (and values per-token), plus a Hadamard rotation on the cache, which restored long-context recall while keeping the 4-bit memory savings. The lesson: keys and values are different distributions and must be quantized differently, and the failure mode of getting it wrong hides from short-prompt evals.

### 9. The calibration set that quietly cost four points

A team quantized a general-purpose model to W4A16 using a stock calibration set of generic web text, validated it on a general benchmark where it looked fine, and shipped it into a product that was almost entirely code completion. In production the quantized model was noticeably worse at code than the FP16 original — more syntax errors, worse multi-line completions — even though the general eval showed almost no regression. The wrong hypothesis was that 4-bit "just isn't good enough for code." The actual root cause was the calibration distribution: the activation statistics of code differ from prose, so the scales and (where used) the GPTQ Hessian estimates were fit to the wrong distribution, leaving the code-relevant channels under-served. Re-quantizing with a code-heavy calibration set drawn from the deployment distribution recovered roughly four points on the code eval at zero runtime cost. The lesson: below the comfortable regime, calibration data is a hyperparameter, not boilerplate — calibrate on what you serve, and validate on the domain you serve, not on a generic benchmark that averages your real workload away.

### 10. The FP4 training run that flatlined

A research team attempting low-precision pre-training quantized both weights and gradients to FP4 with plain round-to-nearest and watched the loss curve stall well above the FP8 baseline — the model was learning, but slowly and to a worse final loss. The wrong hypothesis was that FP4 simply lacks the precision to train at all. The actual root cause was rounding: with round-to-nearest, every weight update smaller than half an FP4 step rounded to zero, and since most updates late in training are small, the model was discarding the majority of its learning signal each step. Switching to stochastic rounding for the FP4 updates, plus a two-level (per-tensor and per-block) scaling scheme to keep tensors in range, closed most of the gap to the FP8 baseline, with the small remaining difference not showing up on downstream evals — mirroring the published NVFP4 pre-training results. The lesson: low-precision *training* fails for reasons low-precision *inference* never encounters — accumulated rounding bias — and the fix is in the rounding rule and the scaling, not in adding bits back.

## A decision framework: which frontier technique, when

The methods in this article are not competitors so much as answers to different questions. The figure routes the four questions that actually determine the choice — target bit-width, whether you must quantize activations, your retrain budget, and your target silicon — to a concrete method.

![Four questions — target bits, activations or weights, retrain budget, target silicon — route you to one method](/imgs/blogs/past-4-bit-wall-frontier-llm-quantization-10.webp)

Read the tree top-down. If you need **4-bit weight-only for memory-bound serving**, you do not need anything in this article past Section 7 — use GPTQ or AWQ with the Marlin or Machete kernel and stop. If you need **W4A4** because you want the low-precision activation math path, you need rotations: start with QuaRot's fixed Hadamard and upgrade to SpinQuant's learned rotation if you need the extra points. If you need **2-bit or below** to fit on hardware that otherwise cannot hold the model, use codebook methods — QuIP#, AQLM, or VPTQ — and accept the decode-speed cost. If you **control training and ship at scale**, native BitNet ternary is the energy-optimal endpoint. And if you are on **Blackwell**, quantize to NVFP4 so the work lands on the native FP4 tensor cores rather than an emulated integer path.

The same logic as a strategy table:

| Situation | Reach for | Why | Main cost |
| --- | --- | --- | --- |
| 4-bit weights, GPU, memory-bound | GPTQ / AWQ + Marlin/Machete | Solved, fast, well-supported | None worth mentioning |
| Need 4-bit *activations* (W4A4) | QuaRot, then SpinQuant | Rotations erase activation outliers | Online Hadamards at runtime |
| 2-bit or below to fit memory | QuIP# / AQLM / VPTQ | Codebooks pack bits where weights live | Decode is gather-bound, slower |
| 1.58-bit, you control training | BitNet b1.58 native QAT | Multiply-free forward, ~10x energy | Full pre-training cost |
| Blackwell hardware | NVFP4 (+ rotation) | Native FP4 tensor cores, ~2.3x over INT4 | Requires Blackwell |
| Long context, memory-bound on KV | Per-channel key quant + rotation | Keys have per-channel outliers | Asymmetric kernel complexity |

### Reach for the frontier when

- **Memory is the hard constraint** and 4-bit weight-only does not fit — going to 2-bit codebooks can be the difference between fitting on the hardware you have and not running at all.
- **You need low-precision activation math** (W4A4) for throughput on hardware where activation bit-width gates the fast path — rotations make this regime survivable.
- **You are on Blackwell** and still running an INT4 software path — switching to NVFP4 is free accuracy and free speed relative to emulation.
- **You serve at enormous scale and control training** — native low-bit's per-inference energy savings eventually dwarf the one-time training cost.
- **Long context dominates your memory** — KV-cache quantization with the right per-channel handling buys 2-4x context per byte.

### Skip the frontier when

- **4-bit weight-only already meets your accuracy and memory targets** — going lower adds risk and complexity for savings you do not need. The comfortable recipe is comfortable for a reason.
- **You are compute-bound at high batch** — lower weight bit-width will not speed you up and weight-only schemes can slow you down; reach for a hardware-native format (FP8/NVFP4), not more aggressive INT bits.
- **You cannot run a real accuracy regression suite** — aggressive quantization without task evals at production context lengths is how a collapsed long-context recall ships to users undetected.
- **You would have to train from scratch and lack the scale to justify it** — native low-bit is not retrofittable; for an off-the-shelf checkpoint, PTQ is your only option.
- **The model is small enough that it already fits comfortably** — quantization's whole value is relieving a memory or bandwidth constraint; with no constraint, you are adding risk for nothing.

## Further reading

- [Quantization in LLMs: the complete practical guide](/blog/machine-learning/large-language-model/quantization-in-llm) — the foundational methods (GPTQ, AWQ, GGUF, bitsandbytes, FP8) this article builds on.
- [INT8 vs FP16 vs INT4: quantization tradeoffs for edge inference](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) — the edge/Jetson-flavored sibling, with SmoothQuant, SpinQuant, and a concrete edge recipe.
- [BitNet b1.58 2B4T technical report](/blog/paper-reading/large-language-model/bitnet-b1-58-2b4t-technical-report) — a close read of the native ternary model behind Section 5.
- [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) — the full picture behind Section 8's KV-cache wrinkle.
- **QuaRot** (Ashkboos et al., 2024), **SpinQuant** (Liu et al., 2024) — the rotation papers.
- **QuIP#** (Tseng et al., 2024), **AQLM** (Egiazarian et al., 2024) — the codebook papers.
- **Marlin** (IST-DASLab) and **Machete** (vLLM/Neural Magic) — the mixed-precision GEMM kernels that make any of this fast.
