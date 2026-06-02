---
title: "TurboQuant: Data-Free KV Cache Quantization, From Random Rotations to vLLM"
date: "2026-06-02"
publishDate: "2026-06-02"
description: "A deep, intuition-first walkthrough of TurboQuant — how random rotations, Lloyd-Max codebooks, and a one-bit JL residual compress the KV cache to 2-3 bits without calibration data, and how the 0xSero/turboquant repo wires it into vLLM."
tags: ["kv-cache", "quantization", "vllm", "llm-inference", "vector-quantization", "long-context", "gpu-memory", "triton", "open-source", "lloyd-max", "johnson-lindenstrauss"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

There is a quiet lie in the way most of us reason about LLM memory. We open a model card, read "27B parameters", multiply by two bytes for FP16, and conclude the model needs ~54 GB of VRAM. That number is real, but it is the *static* cost — the part that never moves. The part that actually decides whether your 30k-token agent request fits on the GPU is the **KV cache**, and that cost is dynamic, grows linearly with context length, and is almost never the thing people budget for. By the time you are serving 100k-token contexts to a handful of concurrent users, the KV cache is no longer a footnote next to the weights. It *is* the bill.

[TurboQuant](https://github.com/0xSero/turboquant) — the open-source repo by `0xSero` implementing the ICLR 2026 paper [*TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni) — is an attempt to make that bill smaller without making the model dumber. It compresses cached **keys to 3 bits** and **values to 2-4 bits**, freeing roughly **30-77%** of KV memory depending on the model, while keeping attention scores statistically faithful. The interesting part is *how*: not a learned codebook trained on your data, not a per-model calibration pass, but a chain of three ideas — a **random orthogonal rotation**, a **Lloyd-Max optimal scalar quantizer**, and a **one-bit Johnson-Lindenstrauss residual** — that together give you an *unbiased estimate of attention scores* with no calibration data at all.

![TurboQuant compress path: a KV vector is rotated, scalar-quantized, sign-corrected, and bit-packed before it touches the cache](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-1.png)

The diagram above is the mental model for the entire post. A key or value vector enters as FP16, gets spun by a fixed random rotation, gets each coordinate snapped to a precomputed codebook, picks up a one-bit sign correction, and finally gets bit-packed into the cache at roughly a quarter of its original size. Everything else in this article is a tour of that pipeline — *why* each box exists, *what* breaks if you remove it, and *how* the repo turns the math into Triton kernels and a vLLM monkey-patch you can actually run.

## Why KV cache quantization is harder than weight quantization

If you have quantized model weights before — GPTQ, AWQ, bitsandbytes — you might assume KV quantization is the same problem with a different tensor. It is not, and the difference is the entire reason TurboQuant needs to be *data-free*.

| Property | Weight quantization | KV cache quantization |
|---|---|---|
| When does the data exist? | Ahead of time, fixed forever | Online, generated token-by-token at inference |
| Can you calibrate on it? | Yes — run a calibration set, fit scales/codebooks | No — every request produces brand-new keys/values you never saw |
| How many times is each value read? | Many (every forward pass) | Few — a key is written once, read by future queries, then evicted |
| What error metric matters? | Output logits / perplexity | **Inner product** `⟨query, key⟩` — the attention score |
| Latency budget to compress | Offline, minutes are fine | Microseconds, on the critical path |

Weight quantization is a *batch* problem: you have all the weights, you can afford an expensive offline fit, and the same quantized weight is reused billions of times so amortizing the cost is trivial. KV quantization is an *online* problem. The keys and values for a request are manufactured during prefill and decode, they are unique to that request, and you must compress them in the time it takes to write them to the cache. There is no calibration set because there is no "set" — there is a stream.

This is why the naive port of weight-quantization tricks fails. Per-tensor scales fit on calibration data assume a distribution you can measure in advance. A learned codebook (product quantization, residual VQ) needs training. Both assume you get a second look at the data. KV cache gives you one look, live, on the hot path. TurboQuant's whole design philosophy is: **make the data distribution predictable enough that you can precompute the optimal quantizer once, offline, and apply it to any vector you have never seen.** The tool that buys that predictability is a random rotation, and we will get there in two sections.

> The senior rule of thumb: weight quantization is a fitting problem; KV quantization is a *streaming estimation* problem. If your method needs to see the data twice, it does not belong in the KV path.

If you want the broader context on why the KV cache dominates long-context serving, the [KV cache deep dive](/blog/machine-learning/large-language-model/kv-cache) on this blog covers the data structure itself, and the [LMCache layer deep dive](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive) covers the orthogonal trick of *reusing* KV across requests. TurboQuant is the third axis: making each cached entry physically smaller.

## The mental model, in three sentences

Before we dissect each stage, hold the whole pipeline in your head as one sentence per phase.

**Compress (write path):** rotate the vector so its coordinates become statistically tame, snap each coordinate to a precomputed codebook level using `b-1` bits, store a single extra sign bit per coordinate to correct the inner-product bias, and pack the result into bytes.

**Store:** the cache now holds 3-bit keys and 2-4-bit values plus per-group scales — roughly 4× smaller than FP16 — and never holds the FP16 originals again after prefill.

**Score (read path):** when a new query arrives, do *not* decompress the keys; instead "sketch" the query through the same projection, combine the codebook estimate with the sign-bit correction, and recover an *unbiased* estimate of the true attention logit.

That asymmetry — compress the keys aggressively, sketch the queries cheaply at read time — is the trick that keeps attention accurate at 3 bits. Now let us earn each piece.

## 1. The memory problem TurboQuant attacks

**Senior rule of thumb: size your deployment by the KV cache at your *target* context length, not by the weights.**

The KV cache stores, for every token and every transformer layer, one key vector and one value vector per attention head. For a model with `L` full-attention layers, `H` KV heads, head dimension `d`, in `p`-byte precision, the cache cost of a single token is:

$$
\text{bytes/token} = 2 \cdot L \cdot H \cdot d \cdot p
$$

The factor of 2 is keys *and* values. For a sequence of `S` tokens across a batch of `B` requests, total KV memory is `B · S · 2 · L · H · d · p`. The thing to notice is that **only `S` is under the user's control and it grows without bound**. Weights are a one-time `O(parameters)` cost; KV is an `O(batch × context)` cost that scales with exactly the dimension your product team keeps asking you to increase.

![Where the GPU memory goes: KV cache grows linearly with context and overtakes weights at long context](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-2.png)

The figure above is the entire motivation in one before/after. On the left, the FP16 KV cache: `2·L·d` bytes per token, linear growth, and a hard OOM wall somewhere past 100k tokens for most single-GPU setups. On the right, TurboQuant's regime: 3-bit keys and 4-bit values shrink each compressed layer by ~4.4×, freeing 30-77% of KV memory and roughly doubling token capacity.

Let us put concrete numbers on it using the repo's own reported benchmarks. On an **RTX 5090 (32 GB)** running Qwen3.5-27B-AWQ at 30k context, TurboQuant reports freeing **30 GB of KV cache across 4 GPUs** and lifting **maximum token capacity from 457k to 914k — exactly 2.0×**, while *gaining* throughput: prefill +5.7%, decode +3.1%. The throughput gain is not magic; it comes from the fact that smaller KV means more requests fit, which means better batch utilization, which more than pays back the compression compute on this hardware.

On an **8× RTX 3090 (24 GB)** cluster running the Qwen3.5-35B-A3B MoE at 131k context, the picture is more sober: **30.9% KV savings per GPU** (233.8 MB saved at max context), a **4.41× compression ratio on the layers it actually compresses**, and **1.45× context extension** (1.4M → 2.0M aggregate tokens). The gap between "4.41× on compressed layers" and "30.9% overall" is the single most important caveat in the whole project, and we will return to it: **TurboQuant only compresses full-attention layers.** MoE and hybrid models that interleave linear-attention or Mamba blocks have a large chunk of KV that TurboQuant leaves untouched.

### Second-order optimization: compression ratio is not memory savings

The trap here is reading "3-bit keys" as "5.3× less KV memory" and stopping. Three things erode that headline:

1. **Values are not keys.** If you keep values at 4 bits (the recommended default), your blended key+value compression is closer to ~3.5-4× than 5.3×.
2. **Scales and zero points cost bits.** Group quantization stores an FP16 scale and zero per group of `g` values. At `g=64`, that is `32 bits / 64 values = 0.5 bits/value` of overhead — small but real.
3. **Uncompressed layers dilute everything.** On a model where 40% of KV lives in linear-attention layers, even infinite compression of the other 60% caps your savings at 60%.

Always compute the *blended* number for *your* model architecture, not the per-layer headline.

### A worked memory budget

Abstract formulas slide off the brain, so let us push real numbers through `bytes/token = 2 · L · H · d · p`. Take a dense model with `L = 48` full-attention layers, `H = 8` KV heads (grouped-query attention), head dimension `d = 128`, in FP16 (`p = 2` bytes):

$$
\text{bytes/token} = 2 \cdot 48 \cdot 8 \cdot 128 \cdot 2 = 196{,}608 \approx 192 \text{ KB/token}
$$

That is **192 KB of KV for every single token**. A 100k-token context is therefore `100{,}000 \times 192 \text{ KB} \approx 18.3 \text{ GB}` — for *one request*. On a 24 GB consumer card with ~14 GB already spent on the (quantized) weights, you have room for *less than one* such request before OOM. This is not a hypothetical; it is the exact wall the RTX 3090 benchmarks were hitting.

Now apply TurboQuant. Keys at 3 bits go from 16 bits to 3 bits — a `16/3 ≈ 5.3×` shrink. Values at 4 bits go `16/4 = 4×`, plus the `0.5 bits/value` group-metadata overhead pushes the effective rate to ~4.5 bits, so `16/4.5 ≈ 3.6×`. Blending keys and values 50/50:

$$
\text{blended} = \frac{2}{\frac{1}{5.3} + \frac{1}{3.6}} \approx 4.3\times
$$

So the 18.3 GB request drops to roughly **4.3 GB** — and suddenly the card holds three or four concurrent 100k-token requests where before it held zero-point-something. That single arithmetic exercise is the entire business case. Notice also that this 4.3× blended number lines up with the repo's reported **4.41× on compressed layers**; the gap to the headline-disappointing **30.9%** on the MoE benchmark is purely the uncompressed linear-attention layers diluting the average. Run this calculation for *your* architecture before you size hardware.

## 2. Random orthogonal rotation: making every coordinate well-behaved

**Senior rule of thumb: you cannot quantize a distribution you cannot predict — so first transform the data into one you can.**

Here is the core obstacle. Key and value vectors coming out of attention layers are *not* nicely behaved. Their energy concentrates in a few dominant coordinates ("massive activations"), they have heavy tails, and the distribution differs per channel, per layer, per model. A single fixed quantization codebook applied to raw coordinates would waste precision on the quiet channels and clip the loud ones. This is exactly why weight-quant methods need calibration: to *measure* the per-channel distribution before fitting.

TurboQuant's first move sidesteps the entire measurement problem with a piece of high-dimensional geometry. Multiply each (unit-normalized) vector by a **fixed random orthogonal matrix** `Π`. An orthogonal rotation preserves lengths and inner products exactly — `⟨Πx, Πy⟩ = ⟨x, y⟩` — so it loses *no information*. But it dramatically changes the *coordinate-wise* distribution.

![What a random rotation buys you: spiky, outlier-prone coordinates become a uniform Beta distribution](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-3.png)

The before/after above is the intuition. Take any unit vector and rotate it by a random orthogonal matrix in `d` dimensions. Each coordinate of the result behaves like a sample from a **Beta distribution** — specifically, the squared coordinate `y_i^2` of a random unit vector follows `Beta(1/2, (d-1)/2)`, and for large `d` the coordinates are *near-independent* and *identically distributed*. The spikes get spread out. The heavy tails get tamed. Most importantly: **every coordinate now follows the same known law, regardless of which model, layer, or token produced the original vector.**

This is the whole game. Once the coordinates are i.i.d. Beta, you can compute *the* optimal scalar quantizer for that one distribution, once, offline, and apply it to every coordinate of every vector forever. No calibration. No per-model fit. The data became predictable by construction.

### The math, briefly

Let `x` be a unit vector (`‖x‖ = 1`) in `ℝ^d`. Apply a random rotation `y = Πx`. Because `Π` is uniformly random over the orthogonal group, `y` is uniformly distributed on the unit sphere `S^{d-1}`. The marginal density of a single coordinate `y_1` on the sphere is:

$$
p(y_1) \propto (1 - y_1^2)^{(d-3)/2}, \quad y_1 \in [-1, 1]
$$

which is symmetric, concentrated near zero, and — crucially — *the same for every coordinate*. As `d` grows, `\sqrt{d} \cdot y_1` converges to a standard Gaussian, but for the finite `d=128` or `d=256` head dimensions used in practice, the repo works directly with the exact Beta-derived density. The norm `‖x‖` is factored out *before* rotation and stored separately (a single FP16 scalar per vector), so the rotation only ever sees unit vectors and the quantizer only ever sees this one fixed distribution.

### Why orthogonal and not just any random matrix

A general random projection (Gaussian matrix) would also spread energy, but it would *not* preserve norms exactly, and reconstruction would require a pseudo-inverse. An *orthogonal* matrix is its own inverse-transpose (`Π^{-1} = Π^T`), so the decompress step is just another matrix multiply by `Π^T` — same cost, exact inversion, no information loss. In the repo, this lives in `rotation.py` as `rotate_forward(x, Pi)` and its transpose counterpart. The matrix `Π` is generated once per head dimension and shared across all layers and requests — it is part of the "codebook", not per-request state.

### Fast rotations: why a dense matmul is not the bottleneck you fear

A reasonable objection: a dense `d × d` rotation is an `O(d²)` matmul per vector, and you are doing it for every key and value on the write path. For `d = 128` that is 16k multiply-adds per vector — is that not exactly the kind of hot-path cost section 1 warned about? In practice it is cheap for two reasons. First, the rotation runs *once per vector at write time*, not per query at read time, so it is amortized over every future attention step that reads the vector. Second, the structure of the random orthogonal family lets you use **fast transforms** instead of a dense matmul: a randomized Hadamard transform (a sign-flip diagonal followed by a Walsh-Hadamard transform) is a near-uniform random rotation that runs in `O(d log d)` with no stored matrix at all. Many production rotation-based quantizers (QuaRot, SpinQuant, and the QJL line of work TurboQuant descends from) use exactly this trick. The repo's `rotation.py` exposes the rotation as a matmul against a stored `Π` for clarity and exactness, but the asymptotic story is that rotation is sub-quadratic and write-path-only — it is never the thing that gates your decode latency.

The deeper reason the rotation is worth its cost: it is the *only* thing standing between you and per-channel calibration. Without it, you would need to measure each channel's distribution (a calibration pass over data you do not have in the online setting) to place quantization levels well. With it, every channel is the same known Beta law, and the one precomputed codebook is provably near-optimal for all of them. You are spending a few thousand FLOPs per vector to *delete an entire calibration stage*. That is one of the best trades in the whole inference stack.

### Second-order optimization: the rotation seed is load-bearing state

Because compress and score both rely on the *same* `Π`, the rotation matrix is not a throwaway — it must be identical at write time and read time. If you regenerate `Π` from a fresh random seed between prefill and decode, every stored key is now rotated by one matrix and queried through another, and your attention scores become noise. The repo pins this; in case study 4 we will see what happens when a careless refactor does not.

## 3. Lloyd-Max optimal scalar quantization

**Senior rule of thumb: equal-width bins are a default, not a choice — put your quantization levels where the probability mass actually is.**

Now that every coordinate follows a known Beta-derived density, we want the *best possible* `(b-1)`-bit scalar quantizer for that density. "Best" here means minimizing expected squared error (MSE) between the original coordinate and its quantized reconstruction. The classical solution is the **Lloyd-Max quantizer**, the 1D special case of k-means: it places quantization levels (centroids) and decision boundaries so that, in expectation, the squared error is minimized for the given source density.

![Uniform vs Lloyd-Max quantization: Lloyd-Max packs more levels where the Beta density is high](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-4.png)

The before/after captures the difference. A *uniform* quantizer cuts the value range `[-1, 1]` into equal-width buckets — simple, but it wastes precious levels out in the tails where the Beta density is almost zero, and starves the high-density region near zero where most coordinates actually land. **Lloyd-Max** instead places levels by probability density: many fine-grained levels packed near zero where coordinates cluster, few coarse levels in the sparse tails. For the same bit budget, this is a strict win in MSE.

The two conditions that define the optimum, iterated to convergence, are:

1. **Nearest-neighbor (centroid) condition:** each value is assigned to the closest centroid. The decision boundary between two centroids `c_i` and `c_{i+1}` sits at their midpoint `(c_i + c_{i+1})/2`.
2. **Centroid (conditional-mean) condition:** each centroid is the conditional mean of the source over its bucket — `c_i = E[y \mid y \in \text{bucket}_i]`.

You alternate these until they stop moving. The beautiful thing: because the source density is *fixed and known* (it is the Beta-derived sphere marginal), this iteration runs **once, offline**, with no data. The result is a small table of decision boundaries and centroids — the **codebook**.

### Data-free codebooks in the repo

This is `codebook.py`. The repo ships **pre-generated codebooks** in `codebooks/` for the common configurations: head dimensions `d ∈ {128, 256}` and bit widths `b ∈ {2, 3, 4}`. Generating them is a fit on the analytic Beta density, not on any model's activations:

The function below is the conceptual shape of `codebook.py` — a Lloyd-Max fit on the analytic Beta source, with no model activations anywhere in sight:

```python
import torch

def lloyd_max_beta(n_levels: int, d: int, iters: int = 200):
    """Fit an n_levels scalar quantizer to the coordinate density of a
    random unit vector in R^d. Fully data-free: the 'data' is sampled
    from the analytic distribution, not from any model."""
    # Sample the exact coordinate density of a random unit vector:
    # draw Gaussians, normalize, take one coordinate.
    g = torch.randn(2_000_000, d)
    coords = (g / g.norm(dim=-1, keepdim=True))[:, 0]  # ~ sphere marginal

    # Initialize centroids on quantiles so empty buckets never happen.
    qs = torch.linspace(0, 1, n_levels + 2)[1:-1]
    centroids = torch.quantile(coords, qs)

    for _ in range(iters):
        boundaries = (centroids[1:] + centroids[:-1]) / 2          # midpoints
        idx = torch.bucketize(coords, boundaries)                  # assign
        for k in range(n_levels):                                  # conditional means
            mask = idx == k
            if mask.any():
                centroids[k] = coords[mask].mean()
    return centroids, boundaries
```

Note `n_levels = 2^(b-1)`, not `2^b`. We spend `b-1` bits on the scalar codebook because the last bit is reserved for the sign-residual trick in the next section. A 3-bit key therefore uses a 4-level (`2^2`) Lloyd-Max codebook plus 1 sign bit.

The shipped codebooks cover the configurations that matter in practice. The table below is the shape of what lives in `codebooks/` — a tiny lookup table, a few hundred bytes each, that you generate once and reuse for every model with that head dimension:

| head_dim `d` | bits `b` | scalar levels `2^(b-1)` | what it's for |
|---|---|---|---|
| 128 | 2 | 2 | aggressive value quant (small heads) |
| 128 | 3 | 4 | default keys (small heads) |
| 128 | 4 | 8 | quality-sensitive values (small heads) |
| 256 | 2 | 2 | aggressive value quant (large heads) |
| 256 | 3 | 4 | default keys (large heads) |
| 256 | 4 | 8 | quality-sensitive values (large heads) |

The crucial property: **this table is independent of the model.** A Qwen, a Llama, and a Mistral with `d = 128` all use the identical `d=128, b=3` codebook, because after rotation their key coordinates all follow the same Beta law. That is the data-free promise made concrete — there is no per-model artifact to fit, version, or ship. If your model uses an off-grid head dimension (say `d = 192`), you generate a codebook for it once with the function above; case study 5 is what happens when you forget to.

The actual quantization at write time is then a single vectorized lookup. From `quantizer.py`:

The MSE quantize step (the paper's Algorithm 1) is faithful to `quantizer.py`'s structure — five lines, each mapping to one box in the mental-model figure:

```python
def quantize_mse(x, Pi, decision_boundaries, bits):
    norms = x.norm(dim=-1, keepdim=False)               # 1) factor out magnitude
    x_unit = x / (norms.unsqueeze(-1) + 1e-10)          # 2) project to unit sphere
    y = rotate_forward(x_unit.float(), Pi)              # 3) rotate -> Beta coords
    indices = torch.searchsorted(decision_boundaries,   # 4) snap to nearest level
                                 y.contiguous())
    packed = _pack_indices(indices, bits)               # 5) bit-pack
    return MSEQuantized(packed=packed, norms=norms)
```

Decompression reverses it exactly: look up the centroid for each index, undo the rotation with `Π^T`, and rescale by the stored norm. The codebook lookup is `searchsorted` against the decision boundaries — an `O(log levels)` binary search that the GPU does in a single fused pass over the tensor.

> Lloyd-Max is k-means that already converged before you showed up. The codebook is a constant of the *geometry*, not a function of your data.

### Second-order optimization: why this matches the information-theoretic bound

The paper proves that this rotate-then-Lloyd-Max scheme achieves a distortion rate **within a constant factor (~2.7×) of the information-theoretic lower bound** across all bit widths. That is the formal version of "you cannot do meaningfully better without per-vector adaptivity." The practical consequence: there is no point hand-tuning a fancier scalar codebook for KV quantization — the gains are bounded by a small constant, and you would be trading that for the calibration cost you just escaped.

## 4. QJL sign bits and the unbiased inner-product estimator

**Senior rule of thumb: in attention, you do not care about reconstructing the key — you care about the inner product `⟨query, key⟩`. Optimize for the metric you actually use.**

Here is the subtle failure mode of any pure-MSE quantizer in attention. MSE quantization minimizes `E‖x - x̂‖²`, the reconstruction error. But attention never reconstructs keys for their own sake — it computes scores `⟨q, k⟩` and softmaxes them. And a quantizer that is unbiased in *reconstruction* can be **biased in inner product**: `E[⟨q, k̂⟩] ≠ ⟨q, k⟩`. The rounding error `k - k̂` is not mean-zero in the direction of `q`, so scores drift systematically. Over a long context, that bias accumulates and the attention distribution shifts — the model starts attending to the wrong tokens.

TurboQuant fixes this with a second stage borrowed from the **Quantized Johnson-Lindenstrauss (QJL)** literature: spend the one remaining bit per coordinate on a *sign-encoded residual* that makes the inner-product estimate provably unbiased.

![The two-stage unbiased estimator: an MSE base layer plus a 1-bit JL residual yields zero bias](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-5.png)

The pipeline above shows the two stages. **Stage 1** is the Lloyd-Max MSE quantizer from the previous section, but at `b-1` bits, producing a base reconstruction `x̂`. **Stage 2** takes the leftover residual `r = x - x̂`, projects it through a random `±1` (or Gaussian) JL matrix `S`, and stores only the **sign** of each projected coordinate — one bit each. At read time, those sign bits, combined with the query sketched through the same `S`, produce a correction term whose expectation exactly cancels the inner-product bias of the base layer.

### The estimator, concretely

From `quantizer.py`, the inner-product (Algorithm 2) quantize step:

The inner-product quantize step (Algorithm 2) layers the QJL residual on top of an MSE base at one fewer bit:

```python
def quantize_prod(x, Pi, boundaries, S, bits):
    base = quantize_mse(x, Pi, boundaries, bits - 1)    # Stage 1: b-1 bits
    x_hat = dequantize_mse(base, Pi, ...)               #   reconstruct base
    residual = x - x_hat                                 # Stage 2: the leftover
    projected = torch.matmul(residual.float(), S.T)     #   JL projection
    signs = _pack_qjl_signs(projected)                  #   keep sign only: 1 bit
    return ProdQuantized(base=base, signs=signs,
                         res_norm=residual.norm(dim=-1))
```

And the scoring path — the part that *never decompresses the key*:

And the scoring path computes the attention logit from compressed keys — the query is the thing we transform, not the key. This mirrors the `attention_score` method in `quantizer.py`:

```python
def attention_score(query, k_mse, signs, S, qjl_scale, res_norm):
    scores_mse = query @ k_mse.T                          # base inner product
    q_sketch   = query @ S.T                              # sketch the QUERY
    scores_qjl = (q_sketch * signs).sum(-1) \
                 * (qjl_scale * res_norm)                 # 1-bit correction
    return scores_mse + scores_qjl                        # unbiased estimate
```

The reason this is unbiased is the QJL identity: for a random sign-projection, `E[sign(S·r) · (S·q)] ∝ ⟨q, r⟩`. So the correction term `⟨q·S, signs⟩` recovers, in expectation, the inner product between the query and the residual that the base layer missed. Add it to the base score and you get `E[\text{estimate}] = ⟨q, k_{\text{mse}}⟩ + ⟨q, r⟩ = ⟨q, k⟩`. The bias is gone.

This is the deepest idea in the project and the one most worth internalizing: **you spend `b-1` bits making the reconstruction good and 1 bit making the *inner product* unbiased.** For attention, the second bit buys more accuracy-per-bit than a third reconstruction bit would, because it targets the metric softmax actually consumes.

The paper quantifies the payoff: **absolute quality neutrality at 3.5 bits/channel** and only **marginal degradation at 2.5 bits/channel** for KV cache quantization — numbers that a pure-MSE quantizer at the same bit budget cannot match because its scores are biased.

### The variance, and why the base layer matters so much

Unbiasedness alone is not enough — an estimator can be unbiased and still useless if its variance is large. A coin flip is an unbiased estimator of 0.5, but you would not bet on a single flip. The reason TurboQuant's estimator is *both* unbiased *and* low-variance is the interplay between the two stages, and it is worth making explicit because it is the crux of why the method works at 3 bits.

The variance of a one-bit JL inner-product estimate scales with the **magnitude of the vector being sketched**. If you sketched the *whole* key with one bit per dimension (pure QJL), the variance would be proportional to `‖k‖²` — large, requiring many sketch dimensions to average down. TurboQuant instead sketches only the residual `r = x - x̂`, whose norm `‖r‖` is small precisely because the Lloyd-Max base layer already captured most of the vector. The variance of the correction term scales with `‖r‖²`, not `‖k‖²`, so it is smaller by exactly the factor the base layer's quality buys you.

This is the elegant part: the `b-1` reconstruction bits and the 1 sketch bit are not two independent tricks bolted together — they *cooperate*. Better base reconstruction shrinks the residual, which shrinks the sketch variance, which tightens the score estimate. Spend more bits on the base and the whole estimator gets sharper. The single sign bit is doing high-leverage work *only because* it operates on a small residual. Strip out the base layer and the same sign bit would give you an unbiased but noisy estimate; strip out the sign bit and the same base layer would give you a low-variance but *biased* estimate. You need both, and the repo's `bits - 1` split is the mechanism that allocates between them.

There is a clean way to see why the bias matters more than it first appears. Over a context of `S` tokens, the softmax normalizes across all `S` attention logits. A *bias* in each logit does not cancel under softmax — it systematically reweights the distribution, and the effect compounds with context length, which is exactly the regime (long context) where you deploy KV compression. A *zero-mean* error, by contrast, partially averages out across the many tokens in the softmax. So converting bias into variance — which is precisely what the QJL residual does — is the single highest-value move for long-context fidelity. This is the mechanistic reason behind the paper's headline that 3.5 bits/channel is quality-neutral: the bits are spent where the softmax is most sensitive.

### Second-order optimization: the residual norm is the trust signal

Notice the correction is scaled by `res_norm = ‖r‖`. When the base layer already nails the vector, the residual is tiny, the correction contributes almost nothing, and you spent one bit on near-zero. When the base layer struggles, the residual is large and the sign-bit correction does real work. The estimator automatically allocates its one bit of "attention" to the vectors that needed it. You do not tune this; it falls out of the math.

## 5. Asymmetric attention scoring

**Senior rule of thumb: compress the thing you store many of (keys), and pay the transform cost on the thing you have few of (the live query).**

Step back and look at what the scoring code above does *not* do: it never calls `dequantize` on the keys. This is the asymmetry that makes 3-bit keys viable. In a batch decode step you have **one query vector per request** but **thousands of cached key vectors** to score against. Decompressing all those keys back to FP16 every step would obliterate the memory savings (you would materialize the full FP16 cache transiently) and burn bandwidth. So TurboQuant flips it: keep the keys compressed, and instead *sketch the query* into the compressed representation's space.

![Asymmetric attention scoring: the query is sketched at runtime so cached keys never leave their 3-bit form](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-6.png)

The dataflow graph above traces both score components for a single (query, key-store) pair. The query feeds two paths: a base path `⟨q, k_mse⟩` that reads the compressed centroids directly, and a sketch path `q·S^T` that, combined with the stored sign bits, yields the QJL correction `⟨q·S, signs⟩ · scale`. The two are summed into the final attention logit. The key store — sitting on the left in its 3-bit-plus-signs form — is *read*, never *expanded*.

This is structurally identical to how asymmetric distance computation works in product quantization for nearest-neighbor search: you quantize the database vectors once and compute distances by table lookup against an un-quantized query. TurboQuant is, in a real sense, **product-quantization for attention**, with the extra QJL stage to make the "distance" (here, inner product) unbiased rather than just low-MSE. The paper explicitly reports superior nearest-neighbor performance versus classical PQ, which is the same machinery viewed through a retrieval lens.

### Where the compute actually goes

The cost of asymmetric scoring is the query sketch `q·S^T`, which is one small matmul *per decode step per request* — cheap, because there is one query. The keys contribute only lookups and sign-dot-products, which the repo fuses into Triton kernels (`triton_kernels.py` ships **three fused kernels** for the decode-time scoring). Compare this to the alternative of dequantizing keys: that would be `O(num_keys × d)` of FP16 materialization every step, exactly the memory traffic you were trying to avoid.

The decode-step scoring shape is one query sketch against many compressed-key lookups. The query transform is `O(d²)` once per request; key scoring is `O(n_keys)` lookups with no dequant:

```python
q_sketch = query @ S.T                       # 1x  : small, per request
logits = compressed_kv.score(query, q_sketch)  # nx : fused Triton, no dequant
attn = torch.softmax(logits, dim=-1)
```

### Second-order optimization: the hybrid decode caveat

Honesty matters here, and the repo is honest about it: the *fully fused* decode path is the goal, but the current **hybrid decode path dequantizes all history to FP32 per step** in some configurations, trading compute for a simpler integration. That means on the decode path you may pay a dequant cost that the asymmetric design was meant to avoid — until the fused kernels are wired in everywhere. This is the difference between the algorithm's asymptotics and the repo's current engineering state, and it shows up directly in case study 3. When you benchmark, measure decode latency specifically, not just memory.

## 6. Values: group quantization and bit-packing

**Senior rule of thumb: keys and values are not symmetric — keys feed inner products, values feed weighted sums, so they deserve different bit budgets.**

So far we have mostly talked about keys, because keys are what attention scores are computed *from* and therefore where the unbiased-inner-product machinery matters. **Values** are different. A value vector is multiplied by the (already-computed) softmax weights and summed — there is no inner product to keep unbiased, just a weighted average to keep accurate. For values, TurboQuant uses a more conventional but well-tuned **group quantization**: split each value vector into groups of `g` channels, and store a per-group **scale** and **zero point** plus the low-bit codes.

![Value bit-packing layout: four 2-bit value codes share one byte, with one scale and zero per group](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-7.png)

The layout above is the physical memory picture. Four 2-bit value codes pack into a single byte (`4 × 2 = 8` bits), or two 4-bit codes per byte. Alongside each group of (here) 64 values sits one FP16 scale and one FP16 zero point used to dequantize: `value ≈ scale × code + zero`. This is the same affine group-quant you have seen in weight quantization, and it lives in `kv_cache.py`.

The bit-packing matters more than it looks. GPUs address memory in bytes; storing 2-bit codes naively in int8 wastes 75% of the space and defeats the entire purpose. The repo packs four 2-bit codes per byte (and two 4-bit codes per byte) with explicit shift-and-mask, so the on-disk size is genuinely 2 bits/value, not 8. Unpacking is a fused operation in the scoring kernels.

Bit-packing for 2-bit codes puts four values per byte with explicit shift-and-mask. This is the shape from `kv_cache.py`:

```python
def pack_2bit(codes):  # codes in {0,1,2,3}, shape [..., 4k]
    c = codes.reshape(*codes.shape[:-1], -1, 4)
    packed = (c[..., 0]
              | (c[..., 1] << 2)
              | (c[..., 2] << 4)
              | (c[..., 3] << 6)).to(torch.uint8)
    return packed                                  # 1 byte holds 4 values

def unpack_2bit(packed):
    return torch.stack([(packed >> s) & 0b11
                        for s in (0, 2, 4, 6)], dim=-1)
```

### The 2-bit value bottleneck

Here is the most important quality fact in the whole repo, and the reason the recommended default is **4-bit values, not 2-bit**. Measured cosine similarity between original and reconstructed vectors at `head_dim = 256`:

| Tensor | Bits | Cosine similarity | Verdict |
|---|---|---|---|
| Keys | 3-bit | **1.000** | Near-lossless — the rotation+QJL machinery shines |
| Values | 2-bit | **0.940** | The quality floor — visible degradation on hard tasks |
| Values | 4-bit | **0.997** | Recommended default for anything quality-sensitive |

![The bit budget: 3-bit keys are near-lossless while 2-bit values are the quality floor](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-9.png)

The matrix above is the decision table. Three-bit keys come back at cosine similarity **1.000** — effectively lossless, because the rotation + Lloyd-Max + QJL stack is purpose-built for the inner products keys feed. Two-bit values come back at **0.940**, and that 6% angular error is enough to matter on long-context reasoning and precise retrieval. Bumping values to 4 bits recovers **0.997** at the cost of half the value compression. The blanket guidance: **use 3-bit keys and 4-bit values as the default; only drop to 2-bit values when you have measured that your workload tolerates it.**

### Second-order optimization: group size is a quality/overhead dial

Smaller groups (`g=32`) give finer scales and better accuracy but more scale/zero overhead; larger groups (`g=128`) save overhead but let one outlier channel stretch the scale for 128 neighbors. At `g=64` with FP16 metadata, overhead is `0.5 bits/value`. If you are at 2-bit values and fighting for quality, shrinking the group is often a better lever than adding a third value bit — measure both.

This is the same family of tradeoffs covered in the general [LLM quantization guide](/blog/machine-learning/large-language-model/quantization-in-llm) and the edge-focused [INT8/FP16/INT4 tradeoffs post](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) — TurboQuant's value path is essentially affine group-quant, and the lessons transfer directly.

## How TurboQuant compares to other KV cache quantizers

**Senior rule of thumb: the KV-quant design space is defined by two questions — do you need calibration data, and do you optimize for reconstruction or for inner product?**

TurboQuant is not the first KV cache quantizer, and understanding where it sits clarifies *why* its specific choices matter. The field roughly splits along the two axes above.

| Method | Calibration | Optimizes for | Key idea | Tradeoff |
|---|---|---|---|---|
| Per-token INT8/INT4 | None | Reconstruction | Naive affine quant per token | Cheap, but outlier channels wreck low-bit quality |
| KIVI (per-channel keys) | None | Reconstruction | Quantize keys per-channel, values per-token | Strong at 2-bit, but per-channel needs layout gymnastics |
| KVQuant | Calibration pass | Reconstruction | Non-uniform datatypes + outlier handling fit offline | Excellent quality, but needs calibration data |
| QJL | None | Inner product | 1-bit JL sketch of keys for unbiased scores | Data-free unbiased scores; TurboQuant's direct ancestor |
| **TurboQuant** | **None** | **Inner product** | **Rotation + Lloyd-Max + 1-bit JL residual** | **Data-free, unbiased, near information-theoretic optimal** |

The lineage worth tracing is **QJL → TurboQuant**. QJL established that you can get *unbiased* attention scores from a one-bit Johnson-Lindenstrauss sketch with no calibration. But a pure 1-bit sketch throws away a lot of magnitude information — its variance is high, so you need many sketch dimensions to get a tight score estimate. TurboQuant's insight is to put a **good MSE base layer underneath** the sketch: spend `b-1` bits on a Lloyd-Max reconstruction that captures most of the vector, then use the one JL bit only on the small residual the base layer missed. Because the residual is small, the high-variance sketch is applied to a small quantity, and the overall variance collapses. You get QJL's unbiasedness with a fraction of the variance, at a controllable bit budget.

The contrast with **KVQuant** is the calibration axis. KVQuant produces beautiful low-bit quality, but it does so by fitting non-uniform datatypes and outlier masks on a calibration corpus *offline*. That is fine when you control the model and can run the calibration, but it is exactly the assumption the online KV setting violates. TurboQuant deliberately gives up the per-model adaptivity KVQuant exploits, and recovers most of it for free through the rotation — which makes every model's coordinates look like the same Beta distribution, so the *one* precomputed codebook is near-optimal everywhere. The paper's ~2.7× constant-factor gap to the lower bound is the formal statement that "you did not lose much by going data-free."

Versus the **naive per-token INT4** baseline that ships in many engines, the difference is starkest at low bits. Naive affine quantization has no defense against outlier channels: one massive activation stretches the scale and starves every other channel of resolution. The rotation is precisely the outlier defense — it spreads any single channel's energy across all dimensions before quantizing, so there are no outlier channels left to stretch the scale. This is the same reason rotation-based *weight* quantizers (QuaRot, SpinQuant) beat naive weight quantization at 4 bits, applied to the KV side.

The honest competitive caveat: for models you fully control and can calibrate, a calibrated method may edge out TurboQuant on absolute quality at a fixed bit budget. TurboQuant's argument is not "always best quality" — it is "best quality *available without calibration*, which is the only regime the online KV cache actually lives in." If you can calibrate and you want the last fraction of a point, look at the calibrated methods. If you are on a stream — which, again, is what KV is — data-free is not a preference, it is a constraint, and TurboQuant is built for it.

## 7. The vLLM integration

**Senior rule of thumb: a compression algorithm with no serving integration is a research artifact; the integration is where the wins or the disappointments actually happen.**

The algorithm is half the repo. The other half is making it run inside [vLLM](/blog/machine-learning/large-language-model/vllm-inference) without forking the engine. TurboQuant does this by **monkey-patching** the attention backend — `integration/vllm.py` intercepts the points where vLLM writes and reads KV — rather than maintaining a hard fork of vLLM internals. The repo targets specific versions (PyTorch 2.10, vLLM 0.18.0, CUDA 12.8) precisely because monkey-patching is version-fragile; the patched functions must match the engine's current signatures.

![vLLM integration lifecycle: prefill writes a normal cache, TurboQuant compresses it, then frees the baseline before decode](/imgs/blogs/turboquant-kv-cache-quantization-deep-dive-8.png)

The lifecycle above is the integration in five beats. A request arrives. **Prefill** runs with vLLM's normal paged FP16 KV allocation — TurboQuant does *not* yet compress during prefill, so prefill memory is unchanged. Once the prefill KV is ready, TurboQuant **compresses** it (rotate + quantize) into the bit-packed store. Then the crucial step: **free the baseline** FP16 cache, releasing the memory back to vLLM's allocator. Finally **decode** proceeds, reading from the compressed store one token at a time and emitting tokens.

That "free-after-prefill" step is what produces the headline memory savings: for the long-context regime where prefill dominates the KV, the FP16 copy lives only transiently, and steady-state memory is the compressed size. It is also why the *prefill* throughput can go *up* (+5.7% on the RTX 5090 numbers): freeing KV lets more requests co-reside, improving batch packing.

### The honest limitations

The repo's own limitation list is unusually candid, and you should read it before deploying:

1. **Full-attention layers only.** Linear-attention and Mamba layers are left uncompressed. On hybrid/MoE architectures this is *the* reason overall savings (~30%) fall far short of the per-layer compression (4.41×).
2. **Prefill uses standard paged allocation.** The FP16 cache is materialized during prefill, so peak prefill memory is *not* reduced — only steady-state decode memory is. If your OOM happens *during* prefill of a giant context, TurboQuant does not save you there.
3. **Hybrid decode dequantizes history to FP32 per step** in the current path — the fused Triton kernels exist (`triton_kernels.py`) but are not deployed across every decode path yet. This is a compute tax on decode latency.
4. **2-bit values are a quality bottleneck** (cos sim 0.940) — covered above; default to 4-bit.

Enabling TurboQuant in a vLLM run touches one conceptual config surface — patch the engine, then construct your `LLM` as usual and the KV path is compressed transparently:

```python
from turboquant.integration.vllm import patch_vllm

patch_vllm(
    key_bits=3,            # near-lossless
    value_bits=4,          # recommended default; 2 only if measured-safe
    group_size=64,         # value group-quant granularity
    free_after_prefill=True,
)
```

### Validating the claims yourself

One thing that sets this repo apart is that it ships its own adversarial validation. The test suite is the documentation:

```bash
pip install -e .
python validate_paper.py          # 9 tests against the paper's theorems
python audit_claims.py            # adversarial: tries to break the claims
python -m pytest test_modular.py  # 19 architecture/integration tests
python -m pytest test_turboquant.py  # 7 core quantizer tests
CUDA_VISIBLE_DEVICES=0,1,4,6 python proof.py  # A/B baseline benchmark
```

`validate_paper.py` checks the nine theoretical properties (unbiasedness, distortion bounds, rotation invariance). `audit_claims.py` is the interesting one — it is written to *attack* the performance claims rather than confirm them, which is exactly the posture you want from a compression library before you trust it with production traffic. All 35 unit tests pass on the maintained configurations. **Run them on your hardware before believing any number in a README — including the ones in this post.**

## Performance engineering: the three Triton kernels

**Senior rule of thumb: a quantization scheme lives or dies in the decode kernel, because decode is memory-bound and you only win if the compressed read is genuinely cheaper than the FP16 read.**

The algorithm could be perfectly unbiased and still lose in production if the implementation materializes intermediate FP32 tensors and saturates memory bandwidth. This is why `triton_kernels.py` ships **three fused kernels** aimed at the decode-time scoring path. Understanding what they fuse tells you where the wins come from — and where the current "hybrid" path leaves wins on the table.

The naive, un-fused decode step would look like this: read packed codes from HBM, unpack them to int8, look up centroids to get FP16 keys, undo the rotation, then run a standard FP16 attention matmul against the query. Every one of those arrows is a separate kernel launch and a separate round-trip to HBM, and the materialized FP16 keys are exactly the memory traffic compression was supposed to avoid. You would shrink storage and *grow* bandwidth — the worst of both worlds.

A fused kernel collapses that chain. It reads the packed codes once, and in registers/shared memory it unpacks, looks up the centroid, applies the query sketch, accumulates the QJL sign correction, and emits the logit — never writing the intermediate FP16 key back to global memory. The three kernels in the repo correspond to the three pieces of work that must happen together for this to pay off:

1. **Unpack-and-score for the MSE base** — reads packed 3-bit key codes, gathers centroids, and computes `⟨q, k_mse⟩` without materializing `k_mse` in HBM.
2. **QJL sign-correction** — reads the packed sign bits, dots them with the pre-sketched query `q·Sᵀ`, and scales by `qjl_scale · ‖r‖` to produce the unbiased correction term.
3. **Value dequant-and-weighted-sum** — unpacks the 2/4-bit value codes, applies the per-group affine `scale · code + zero`, and accumulates the softmax-weighted value sum.

When all three are on the active path, decode reads ~4× less from HBM and the compute (unpack, lookup, dot) is cheap arithmetic that hides under the memory latency you just reduced. This is why the RTX 5090 numbers show decode throughput going *up* (+3.1%) rather than down.

The honest caveat, repeated because it matters: the current **hybrid decode path does not always route through these kernels**. In configurations where it falls back to "dequantize all history to FP32 per step", you pay the bandwidth you were trying to save, and decode latency can *regress* (see case study 3). The kernels exist; wiring them into every vLLM attention backend variant is ongoing engineering. The practical takeaway: **when you benchmark, confirm via a profiler that the fused kernels are actually firing on your backend** — do not assume the asymptotics from the math.

> A fused decode kernel is the whole ballgame for KV quantization. Without it you have a smaller cache that reads slower; with it you have a smaller cache that reads faster. The gap between those two outcomes is one kernel launch boundary.

## Real-world use cases where TurboQuant earns its keep

Now the part you actually came for: where does this help in practice, and where is it a trap?

### Use case 1: Long-context RAG and agentic pipelines

The single best fit. Retrieval-augmented and agentic workloads stuff tens of thousands of tokens — retrieved documents, tool outputs, scratchpad reasoning — into the prompt, and most of that context is *keys and values that get attended to but rarely regenerated*. This is the regime where KV dominates memory and where 3-bit keys at cosine 1.000 cost you essentially nothing. If your agent times out at 60k tokens because the KV cache OOMs, TurboQuant's 1.45-2.0× context extension is the difference between "fits" and "doesn't". The paper's 5/5 needle-in-haystack retrieval at all context lengths is precisely the property a long-context agent needs.

Make it concrete. Picture a coding agent that loads a 40k-token slice of a repository, a 15k-token conversation history, and 10k tokens of tool output from test runs — 65k tokens of context, almost all of it read-mostly. On a 24 GB card running a quantized 14B model, that single request's FP16 KV can be 10-12 GB, leaving no room for a second concurrent agent and pushing you toward a bigger GPU. With 3-bit keys and 4-bit values the same request drops to ~3 GB, and now three agents share the card. The keys — which is what the model uses to *find* the relevant function or the right line of the stack trace — are near-lossless, so retrieval quality holds. This is the exact shape of workload the design targets: huge read-mostly context, retrieval-dominated attention, memory as the binding constraint. If your product is agents or RAG over long documents, this is not a marginal optimization; it is a tier change in what hardware can serve your traffic.

### Use case 2: Multi-tenant serving density

If you run a serving fleet, your unit economics are tokens-per-GPU-per-second, and concurrency is gated by how many requests' KV caches fit simultaneously. Freeing 30% of KV memory means ~30-40% more concurrent requests on the same hardware, which on the RTX 5090 numbers came with a *throughput gain*, not a loss, because better batch packing outran the compression overhead. For a SaaS inference provider, this is a direct margin lever — more tenants per GPU at equal latency.

### Use case 3: Consumer-GPU and local inference

The 8× RTX 3090 and single RTX 5090 benchmarks are not accidental — TurboQuant is explicitly validated on consumer hardware where 24-32 GB is the hard ceiling. Doubling token capacity (457k → 914k on the 5090) is what lets a hobbyist or a small team run a 100k-context workload on a card that otherwise tops out at 50k. If you are building local-first AI tooling, this widens the set of models and contexts you can serve without renting an H100.

### Use case 4: The MoE caveat — when *not* to expect miracles

Mixture-of-experts and hybrid models that interleave linear-attention or Mamba blocks are where expectations need calibrating. TurboQuant compresses only full-attention layers, so on the Qwen3.5-35B-A3B MoE the per-GPU savings were 30.9%, not the 4.41× the compressed layers achieved. That is still a real win — 1.45× context — but if you read "4.41× compression" and budgeted for it on an MoE, you would be off by a factor of three. **Compute your blended savings from your architecture's full-attention-layer fraction before committing.**

## Case studies from production

These are scenarios — some drawn from the repo's own limitation notes, some the kind of incident any team integrating an online KV quantizer will recognize. Concrete because vague case studies teach nothing.

### 1. The 2-bit value regression on a math benchmark

A team enabled TurboQuant with `key_bits=3, value_bits=2` to maximize savings, ran their standard chat evals, saw no regression, and shipped. A week later, accuracy on a multi-step arithmetic benchmark had quietly dropped ~4 points. The wrong first hypothesis was "the rotation is lossy." The actual root cause: **2-bit values at cosine similarity 0.940** — the 6% angular error in value vectors corrupted the precise numeric reasoning that arithmetic chains depend on, even though it was invisible in free-form chat. The fix was a one-line change to `value_bits=4` (cosine 0.997), which restored accuracy at the cost of halving value compression. The lesson: **chat evals do not exercise value precision; test on the hardest precision-sensitive task you have before trusting 2-bit values.**

### 2. The MoE that "only" saved 30%

An infra team benchmarked TurboQuant on a dense 27B model, measured ~4× KV savings, and projected the same onto their production MoE. In production the savings came in at 30.9% and someone filed a bug. There was no bug. The MoE interleaves linear-attention layers that TurboQuant **does not compress**, so a large fraction of KV was never touched. The wrong hypothesis was "compression is broken on MoE"; the root cause was the architecture's full-attention-layer fraction. The fix was not code — it was recomputing the projection from the *blended* compressible-layer ratio. The lesson: **per-layer compression ratio and whole-model memory savings are different numbers, and on hybrid architectures the gap is enormous.**

### 3. The decode-latency surprise

A team integrated TurboQuant chasing memory savings, got them, and then noticed p99 decode latency had crept up ~15%. They expected the asymmetric scoring to be free. The root cause was the **hybrid decode path dequantizing history to FP32 per step** instead of using the fused Triton kernels everywhere — the compute the asymmetric design was supposed to avoid was being paid after all, because the fused path was not wired into their configuration. The fix was to ensure the fused scoring kernels (`triton_kernels.py`) were on the active code path for their attention backend. The lesson: **memory savings and latency are independent axes — measure decode latency explicitly, because the algorithm's asymptotics and the repo's current engineering state can diverge.**

### 4. The rotation-seed mismatch

During a refactor, an engineer "cleaned up" the rotation setup so that `Π` was generated lazily inside both the compress and the score code paths — each from a fresh default RNG. Memory looked great. Outputs were garbage: the model emitted fluent nonsense, attending seemingly at random. The wrong hypothesis was "the quantizer is broken"; the actual cause was that **keys were rotated by one `Π` at write time and queried through a *different* `Π` at read time**, so every inner product was computed in a mismatched basis. The fix was to pin `Π` as shared, persisted state generated once per head dimension. The lesson: **the rotation matrix is part of the codebook, not per-call scratch — write-time and read-time `Π` must be byte-for-byte identical.**

### 5. Validating before trusting: the audit_claims catch

Before rolling TurboQuant out, a cautious team ran `audit_claims.py` on their own hardware rather than trusting the README. The adversarial audit flagged that their chosen configuration — an unusual head dimension without a pre-generated codebook in `codebooks/` — was falling back to a suboptimal codebook, eroding the unbiasedness guarantee. The fix was to generate a Lloyd-Max codebook for their specific `d` via `codebook.py` before deploying. The lesson: **the pre-generated codebooks cover `d ∈ {128, 256}` and `b ∈ {2,3,4}`; off-grid configurations need a one-time offline codebook fit, and the audit script is what tells you that, not a crash.**

### 6. Capacity-doubling that paid for itself

A small inference provider on a single RTX 5090 was turning away long-context requests at 457k aggregate tokens. Enabling TurboQuant (3-bit keys, 4-bit values, free-after-prefill) lifted capacity to **914k tokens — 2.0×** — and *prefill throughput rose 5.7%* because the freed KV improved batch packing. The wrong assumption going in was "compression always costs throughput." On this hardware and workload it was throughput-positive: the engine spent less time memory-bound. The lesson: **on memory-bound serving, KV compression can be a throughput win, not just a memory win — because the bottleneck you relieve is the one that was actually limiting you.**

### 7. The prefill OOM that compression did not fix

A team's requests were OOMing and they reached for TurboQuant expecting relief. It did not come — the OOM happened *during prefill of a 200k-token document*, and TurboQuant **uses standard paged FP16 allocation during prefill**, compressing only afterward. Peak prefill memory was unchanged, so the OOM persisted. The wrong hypothesis was "compression isn't working"; the root cause was *when* compression happens in the lifecycle. The fix was chunked prefill to cap peak prefill memory, with TurboQuant handling the steady-state decode savings. The lesson: **TurboQuant reduces steady-state decode KV, not peak prefill KV — match the tool to where your memory pressure actually occurs.**

### 8. The version-pin that broke on a vLLM upgrade

A platform team had TurboQuant running happily on vLLM 0.18.0, then took a routine upgrade to a newer vLLM for an unrelated scheduler fix. After the bump, requests either crashed on startup or, worse, produced subtly wrong outputs. The wrong hypothesis was "the model checkpoint is corrupted." The actual cause was that TurboQuant integrates by **monkey-patching specific vLLM attention-backend functions**, and the upgrade had changed those functions' signatures — so the patch either failed to apply or applied to the wrong call site. The fix was to pin vLLM back to the supported version and treat TurboQuant upgrades and vLLM upgrades as a *coupled* change that must be tested together. The lesson: **monkey-patch integrations are tied to exact upstream versions; a "routine" engine upgrade is a TurboQuant-breaking change until proven otherwise. Pin both, bump both together, re-run the test suite.**

### 9. The needle that survived compression

A retrieval team was nervous about deploying any KV compression on their long-context recall product — the entire value proposition was finding a single fact buried in 131k tokens, and a lossy cache felt like russian roulette. Before trusting it, they ran a needle-in-haystack sweep at every context length with TurboQuant enabled at 3-bit keys / 4-bit values. The result: **5/5 needle retrieval at all lengths**, matching the repo's reported numbers, because keys come back at cosine similarity 1.000 and retrieval is fundamentally a *key* inner-product problem — exactly what the unbiased estimator protects. The would-be disaster never materialized. The lesson: **retrieval accuracy depends on key fidelity, and keys are the part TurboQuant compresses near-losslessly; do not let "lossy compression" anxiety override the measurement — but do run the measurement.**

### 10. The blended-savings spreadsheet that saved a quarter

Before committing TurboQuant to a serving fleet, an infra lead built a one-page model: for each deployed architecture, the fraction of KV in full-attention layers, the chosen key/value bit budget, and the resulting blended savings and concurrency lift. For the dense models it projected ~4× and ~3.5× more concurrency; for the heavy-MoE model it projected only ~1.3× and flagged it as "not worth the integration risk." Capacity planning was done off that sheet rather than off the README's best-case number. When the dense deployments came in within a few percent of projection and the MoE was correctly left alone, the quarter's GPU budget held. The wrong move — the one they avoided — was applying one headline ratio uniformly across a heterogeneous fleet. The lesson: **the unit of compression-planning is the architecture, not the company; a five-minute blended-savings calculation per model is the cheapest risk reduction you will do all quarter.**

## Common misconceptions

A few beliefs that sound right and cost real time when they turn out wrong.

**"3-bit keys must be lossy, so quality drops."** The cosine similarity for 3-bit keys is 1.000 to three decimals. The reason is that the rotation + Lloyd-Max + QJL stack is engineered for the *inner product*, which is the only thing attention reads from keys. Reconstructing the raw key would indeed lose information; recovering the inner product does not, in expectation. Keys are the easy part.

**"It compresses the whole KV cache."** It compresses *full-attention* layers only. On a model that is half linear-attention, half your KV is untouched, and your ceiling is 50% savings no matter how aggressive the bits.

**"Memory savings imply latency savings."** Independent axes. You can save 30% memory and *lose* latency if the fused decode kernels are not on your path. Profile decode separately.

**"It saves prefill memory."** No — prefill uses standard FP16 paged allocation and the compression happens *after* prefill, with the baseline freed before decode. If your OOM is during prefill, reach for chunked prefill, not TurboQuant.

**"Lower bits always means more savings worth taking."** 2-bit values hit cosine 0.940, which is a real quality floor on precision-sensitive tasks. The extra compression over 4-bit values is often not worth the accuracy you trade for it. Default to 4-bit values and only drop lower with measurements in hand.

**"Data-free means lower quality than calibrated methods."** The paper proves the rotate-then-quantize scheme lands within a small constant (~2.7×) of the information-theoretic distortion lower bound. Calibration buys you per-vector adaptivity that this constant already nearly captures — and calibration is *impossible* in the online KV setting anyway. Data-free is not a compromise here; it is the only thing that works on a stream.

## Operational tuning: the knobs that actually matter

When you deploy, four dials decide your quality/memory/latency point. Here is how to set them without flailing.

**`key_bits` — leave at 3.** Keys come back at cosine 1.000 at 3 bits thanks to the unbiased estimator; dropping to 2 saves little (keys are already the small half once values are 4-bit) and starts to bite. Going to 4 buys you essentially nothing measurable. Three is the sweet spot the whole method is tuned around.

**`value_bits` — start at 4, drop to 2 only with evidence.** This is the dial that most affects quality. At 4 bits, values are cosine 0.997 — safe for almost everything. At 2 bits, 0.940 — fine for casual chat, dangerous for math, code, and exact retrieval. The right protocol is: deploy at 4, establish your quality baseline, then A/B 2-bit on your *hardest* eval and only adopt it if the regression is within tolerance. Never start at 2.

**`group_size` — 64 is a good default; shrink it before adding value bits.** If you are quality-constrained at 2-bit values and 4-bit is too expensive, halving the group to 32 often recovers accuracy for less memory than a third value bit, because it gives outlier-prone groups a tighter scale. Larger groups (128) save metadata but let one channel dominate the scale for more neighbors.

**`free_after_prefill` — keep it on.** This is what turns per-layer compression into actual freed memory and improves batch packing. The only reason to turn it off is debugging a correctness issue where you want the FP16 baseline retained for comparison.

A sane rollout sequence: enable `key_bits=3, value_bits=4, group_size=64, free_after_prefill=True`; confirm the fused decode kernels fire via a profiler; run `validate_paper.py` and `audit_claims.py` on your hardware; benchmark *decode latency and quality together*, not just memory; only then consider pushing values lower. Treat every bit you remove as a hypothesis to be tested on your worst-case task, not a free win.

> The discipline that separates a good KV-quant deployment from a quiet quality regression: change one dial at a time, measure on your hardest eval, and never trust a headline ratio you have not reproduced. The defaults exist because someone already paid for the lesson.

## When to reach for TurboQuant — and when not to

### Reach for TurboQuant when:

- **Your KV cache, not your weights, is the binding memory constraint** — long-context, high-concurrency, or both. This is the design center.
- **You serve long-context RAG or agentic workloads** where 3-bit keys at cosine 1.000 are effectively free and context extension is the whole game.
- **You are on a dense, full-attention model** (the compressible fraction is high) and want maximum savings per layer.
- **You are memory-bound on consumer GPUs** (24-32 GB) and need to roughly double token capacity to fit your workload at all.
- **You cannot calibrate** — the workload is too varied, or you ship a model you do not control the activations of. Data-free is the feature.
- **You run multi-tenant serving** and want concurrency density as a margin lever, and you have measured that compression is throughput-neutral-or-positive on your hardware.

### Skip TurboQuant (or proceed with eyes open) when:

- **Your OOM is during prefill**, not decode — prefill is uncompressed, so this tool does not address peak prefill memory. Use chunked prefill instead.
- **Your model is heavily MoE/hybrid** with a large linear-attention or Mamba fraction — recompute blended savings before assuming the per-layer ratio applies; you may be disappointed.
- **You need 2-bit values for the savings to matter but your task is precision-sensitive** (math, code, exact retrieval) — the 0.940 cosine floor will bite; if 4-bit values do not save enough, this is not your tool.
- **You are pinned to a vLLM/PyTorch/CUDA version far from the supported set** — the integration is monkey-patched and version-fragile; an unsupported engine version is a maintenance liability.
- **Your context lengths are short** (a few thousand tokens) — KV is not your bottleneck, the weights are; compression overhead buys you little.
- **You have not run the test suite on your hardware** — `validate_paper.py` and `audit_claims.py` exist for a reason; an off-grid head dimension silently degrades the guarantees.

> The one-line verdict: TurboQuant is the right tool when long-context KV is your memory ceiling, your model is mostly full-attention, and you can live with 4-bit values. It is the wrong tool for prefill OOMs, heavy-MoE architectures, and 2-bit-value precision tasks. Everything else is a measurement.

## Further reading

- **The paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni, ICLR 2026) — the theory behind the rotation, the distortion bounds, and the unbiased inner-product estimator.
- **The repo:** [0xSero/turboquant](https://github.com/0xSero/turboquant) — implementation, pre-generated codebooks, Triton kernels, vLLM integration, and the self-auditing test suite.
- [KV cache, end to end](/blog/machine-learning/large-language-model/kv-cache) — the data structure TurboQuant compresses.
- [LMCache KV cache layer deep dive](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive) — the orthogonal trick of *reusing* KV across requests; pairs well with compressing it.
- [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) and [INT8/FP16/INT4 edge tradeoffs](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) — the general quantization background that TurboQuant's value path builds on.
- [vLLM inference](/blog/machine-learning/large-language-model/vllm-inference) — the serving engine TurboQuant monkey-patches.
