---
title: "Efficient attention and vision transformers for the edge"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Make self-attention and vision transformers fit on-device: the math behind linear attention and FlashAttention, the hybrid ViTs that beat CNNs on a phone, and the code to measure it."
tags:
  [
    "edge-ai",
    "model-optimization",
    "transformers",
    "attention",
    "vision-transformers",
    "flashattention",
    "inference",
    "efficient-ml",
    "mobilevit",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-1.png"
---

The first time I shipped a transformer onto a phone, the demo crashed in the customer's hands. Not the network — the *math*. We had a vision transformer (ViT) classifying full-resolution camera frames, and it ran beautifully on the workstation. On the device it allocated a buffer the size of the entire app's heap, hit the OOM killer, and died. The post-mortem was one line: the self-attention score matrix for a 384×384 input is roughly 2,300 tokens square, and a 2,300×2,300 float matrix per head per layer is tens of megabytes that has to *exist all at once*. Nobody had read the cost model. We had been thinking in FLOPs and parameters, and the thing that killed us was memory that scales as the square of the token count.

That is the whole story of transformers on the edge in miniature. Transformers dominate accuracy — on language, on vision, on audio — but vanilla self-attention is $O(n^2)$ in the number of tokens, in both compute *and* memory, and the edge is exactly where you can least afford a quadratic. A high-resolution image is thousands of tokens. A long context is thousands of tokens. The device has a few megabytes of fast memory and a thermal budget measured in single-digit watts. The naive answer — "just use a smaller model" — leaves accuracy on the table that you do not have to give up. The real answer is that we have learned, over the last several years, how to *change the cost model itself*: how to make attention linear, how to make it memory-efficient without changing a single output value, and how to build vision transformers that borrow the cheap parts of convolutional networks (CNNs) so the transformer only does the work it is uniquely good at.

By the end of this post you will be able to: derive exactly where the $O(n^2 d)$ cost and the $n \times n$ memory come from; reorder the matrix products to get genuinely linear attention and know precisely what that approximation gives up; explain why FlashAttention is faster despite doing the *same* arithmetic (it is a memory-traffic win, not a FLOP win — straight off [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives)); pick a hybrid vision transformer like MobileViT or EfficientViT that beats a strong mobile CNN at the same latency; and measure all of it honestly with real PyTorch. Figure 1 shows the stakes up front — doubling the tokens does not double attention's cost, it quadruples it.

![Side by side comparison showing vanilla attention memory growing quadratically with token count while linear attention grows in proportion to token count](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-1.png)

This is one of the four-lever levers in disguise: efficient attention and efficient ViT architectures are the *efficient-architecture* lever from the series' [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), and it composes with the others — you will still quantize and prune the result. Let us start with the cost we are fighting.

## 1. Where the quadratic actually comes from

Self-attention takes a sequence of $n$ tokens, each represented by three learned projections — queries $Q$, keys $K$, and values $V$, each an $n \times d$ matrix where $d$ is the head dimension. The whole operation is one famous line:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

It is worth slowing down on every matrix in that line, because the cost — and the fix — lives in the *order* you multiply them.

Before the cost, the *meaning*, because it tells you what each approximation can and cannot give up. The entry $S_{ij} = q_i^\top k_j$ measures how much token $i$ should listen to token $j$; the softmax turns each row into a probability distribution over which tokens to attend to; and the weighted sum $\sum_j P_{ij} v_j$ pulls in the values of the attended-to tokens. The power of attention is that this is *content-based* and *all-to-all* — any token can attend to any other based on what they contain, not on fixed positions. That all-to-all reach is exactly what convolutions lack and exactly what costs $O(n^2)$: there are $n^2$ pairs to score. Every efficiency trick in this post is, at heart, a different answer to "which of those $n^2$ pairs do we actually need?"

**Step 1: the scores $S = Q K^\top$.** $Q$ is $n \times d$, $K^\top$ is $d \times n$, so $S$ is $n \times n$. Every entry $S_{ij}$ is a dot product of two $d$-vectors, which is $d$ multiply-adds. There are $n^2$ entries, so this step is $n^2 d$ multiply-adds. And the *result* is an $n \times n$ matrix that has to be stored.

**Step 2: the softmax.** Applied row-wise to $S$. That is $O(n^2)$ work — cheap relative to the matmuls in FLOPs, but it forces the full $n \times n$ matrix to exist in memory at once, because softmax needs the whole row (it normalizes by the row sum). Hold that thought; it is the crux of FlashAttention.

**Step 3: the context $P V$,** where $P = \text{softmax}(S)$ is $n \times n$ and $V$ is $n \times d$. The product is $n \times d$, and it costs $n^2 d$ multiply-adds by the same accounting as step 1.

Add it up. Compute is $O(n^2 d)$, dominated by the two matmuls. Memory for the score/probability matrix is $O(n^2)$ — and that term is per head, per layer. With $h$ heads the score matrix is $h n^2$ values; with $L$ layers, if you keep activations for the backward pass, you multiply again. The $d$ in the cost is small and fixed (often 64 or 128). The $n$ is the one that moves, and it moves *squared*.

#### Worked example: the buffer that crashed the demo

Take a ViT classifier on a 384×384 image with 16×16 patches. That is $24 \times 24 = 576$ patches, plus one class token, so $n = 577$. With 12 heads, the float32 score matrix per layer is $12 \times 577^2 \times 4$ bytes ≈ **16 MB per layer**. That is already a lot, but a ViT-Base has 12 layers, and during inference you can free each layer's scores before the next — so the peak is ~16 MB, survivable. Now push to a 768×768 input (semantic segmentation, document understanding): $n = 48 \times 48 + 1 = 2{,}305$. The per-layer score matrix becomes $12 \times 2305^2 \times 4$ ≈ **255 MB**. On a phone with a 200–300 MB practical heap budget for your process, a single attention layer wants the entire budget. That is the crash. The fix is not a smaller model; it is a cheaper *cost model*.

The compute side scales just as brutally. At $n = 577$ the two attention matmuls are about $2 \times 577^2 \times 64 \times 12 \approx 0.5$ GFLOP per layer; at $n = 2305$ they are about $2 \times 2305^2 \times 64 \times 12 \approx 8.2$ GFLOP per layer — a 16× jump for a 4× increase in resolution-tokens, exactly the quadratic. Multiply by 12 layers and you are at ~100 GFLOP for attention alone, before the feed-forward blocks. A mobile NPU (neural processing unit — the dedicated matrix engine on a modern phone SoC, or system-on-chip) that does a few TOPS will choke on that at any interactive frame rate.

It is worth being precise about *which* term dominates as $n$ grows, because that decides which optimization pays off. A transformer block has two big pieces: attention and the feed-forward network (FFN). The FFN is two linear layers, $d \to 4d \to d$, applied per token, costing $O(n \cdot d^2)$ — *linear* in $n$. Attention costs $O(n^2 d)$ — *quadratic* in $n$. So there is a crossover token count: below it, the FFN dominates and the model is essentially a per-token MLP; above it, attention dominates and the quadratic eats everything. Set the two equal: $n^2 d \approx n \cdot d^2$ gives $n \approx d$. With a model dimension around 768 and head dimension 64, the per-head crossover is low and the per-layer crossover (summing over heads, $d_{\text{model}} = h \cdot d$) lands at roughly $n \approx d_{\text{model}} \approx 768$. The upshot: for short sequences (a sentence, a small image) the FFN is your cost and attention tricks barely help; for long sequences (high-res images, long context) attention is your cost and *this whole post applies*. Always check where your $n$ sits relative to that crossover before optimizing — I have watched engineers spend a week on a clever attention kernel for a model whose actual bottleneck was the FFN.

A second subtlety the FLOP count hides: the $n \times n$ matrix is not just expensive to compute, it is expensive to *move*. Computing $QK^\top$ produces $n^2$ numbers that must be written somewhere, read back for the softmax, written again, read again for the multiply by $V$. On any real chip the cost of those reads and writes — the **memory traffic** — can exceed the cost of the arithmetic itself, especially for the large, low-arithmetic-intensity score matrix. Holding that distinction (compute cost vs memory-traffic cost) is the key that unlocks why FlashAttention works without changing the math, and it is the reason "count the FLOPs" is the wrong instinct here.

There are three honest escape routes, and the rest of this post is each of them: **(1)** approximate the softmax so the cost stops being quadratic at all (linear, windowed, low-rank attention); **(2)** keep the exact math but stop moving the big matrix through slow memory (FlashAttention); **(3)** stop feeding the transformer so many tokens in the first place, and let cheap convolutions do the early heavy lifting (hybrid vision transformers, token merging). They compose. Let us take them in order.

One framing to carry through: each route attacks a different axis of the $n^2$ pairs. Approximation reduces *how faithfully* each pair is scored (linear attention) or *which* pairs are scored at all (windowed, low-rank). FlashAttention changes *how the pairs are moved through memory* without changing what is computed. Token reduction changes *how many tokens exist* so there are fewer pairs to begin with. Because they hit different axes, the best edge stacks layer them — a windowed-attention model whose windows run a flash kernel, token-merged to shrink the grid, then quantized. Keep that mental map and the rest of the post is just filling in each axis with math, code, and measured numbers.

## 2. Linear attention: the associativity trick

Here is the single most important idea in efficient attention, and it is almost embarrassingly simple once you see it. Matrix multiplication is associative. The cost is not.

Forget the softmax for one moment and pretend attention were just $(Q K^\top) V$. You have a choice of where to put the parentheses:

$$
(Q K^\top) V \quad\text{versus}\quad Q (K^\top V)
$$

The two give the *same answer* (associativity), but they cost wildly different amounts. On the left, $Q K^\top$ is $n \times n$ ($n^2 d$ work, $n^2$ memory), then times $V$ is another $n^2 d$. On the right, $K^\top V$ is $(d \times n)(n \times d) = d \times d$ — that costs $n d^2$ and the result is a tiny $d \times d$ matrix — then $Q$ times that $d \times d$ matrix costs $n d^2$. Total: $O(n d^2)$ instead of $O(n^2 d)$. Since $d$ is small and fixed and $n$ is large, $O(n d^2)$ is linear in the sequence length. That is the whole game.

![Side by side comparison of vanilla attention computing the n by n score matrix first versus linear attention reordering the products to build a small d by d state first](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-2.png)

The catch, of course, is the softmax. $\text{softmax}(QK^\top)$ is a *nonlinear* function of the $n \times n$ scores, and you cannot just slide the parentheses past a nonlinearity. So the trick is to replace the softmax similarity with a **kernel** that factorizes. Write generalized attention as

$$
\text{out}_i = \frac{\sum_{j=1}^{n} \text{sim}(q_i, k_j)\, v_j}{\sum_{j=1}^{n} \text{sim}(q_i, k_j)}
$$

where for ordinary attention $\text{sim}(q, k) = \exp(q^\top k / \sqrt{d})$. The insight of Katharopoulos et al. (2020), *Transformers are RNNs*, and of Performer (Choromanski et al., 2021), is: choose a similarity that can be written as an inner product of *feature maps*, $\text{sim}(q, k) = \phi(q)^\top \phi(k)$. Then the numerator becomes

$$
\sum_j \phi(q_i)^\top \phi(k_j)\, v_j = \phi(q_i)^\top \underbrace{\Big(\sum_j \phi(k_j)\, v_j^\top\Big)}_{S \,\in\, \mathbb{R}^{d' \times d}}
$$

The sum $S = \sum_j \phi(k_j) v_j^\top$ does not depend on $i$. You compute it **once** — it is a $d' \times d$ matrix where $d'$ is the feature dimension — and then every query reuses it. Likewise the denominator's normalizer is $\phi(q_i)^\top \sum_j \phi(k_j)$, and $z = \sum_j \phi(k_j)$ is a single $d'$-vector computed once. So the entire attention output is:

$$
\text{out}_i = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}
$$

Building $S$ and $z$ costs $O(n d' d)$ (one pass over the $n$ keys); applying them to all queries costs $O(n d' d)$. No $n \times n$ matrix is ever formed. Linear time, linear memory. The kernel trick is exactly the associativity reorder from Figure 2, made legal across the nonlinearity by choosing $\phi$.

What does $\phi$ look like? The simplest, from the linear-transformer paper, is $\phi(x) = \text{elu}(x) + 1$ — a cheap elementwise feature map that keeps everything positive (so the denominator stays positive and the thing behaves like a similarity). Performer uses random Fourier-style features that *provably approximate* the softmax kernel in expectation: $\phi(x)$ is a random projection followed by $\exp$, and as you add features the approximation tightens. The elementwise version gives up the most fidelity; the Performer version trades extra feature dimensions for a closer match to true softmax.

Performer's trick — the FAVOR+ mechanism — deserves one more sentence of derivation because it shows that linear attention need not be a crude approximation. The softmax kernel can be written as a Gaussian kernel after rescaling: $\exp(q^\top k) = \mathbb{E}_{\omega}\!\big[\,e^{\omega^\top q - \|q\|^2/2}\; e^{\omega^\top k - \|k\|^2/2}\,\big]$ where $\omega$ is drawn from a standard Gaussian. That expectation is itself an inner product of two functions, one of $q$ and one of $k$ — exactly the factorized form we need. Drawing $m$ random $\omega$ vectors and stacking $\phi(x) = \tfrac{1}{\sqrt{m}} \exp(\Omega x - \|x\|^2/2)$ gives an *unbiased* estimate of the softmax similarity whose variance falls as $1/m$. So you can dial the fidelity: more random features ($m$) means closer to exact softmax at higher cost; fewer means cheaper and rougher. The positive-valued exponential features (rather than trigonometric ones) are what keep the estimator stable and the denominator positive — the "+" in FAVOR+. The point for an edge engineer: linear attention is a *spectrum*, from the dirt-cheap $\text{elu}+1$ to a tunable-accuracy random-feature approximation, and you pick the point on that spectrum your accuracy budget allows.

#### Worked example: the d-by-d state never grows with the sequence

Make the memory win concrete. Take $n = 16{,}384$ tokens (a long document, or a 1024×1024 image at 8×8 patches), head dimension $d = 64$, 8 heads, fp16. Vanilla attention's score matrix is $8 \times 16384^2 \times 2$ bytes $\approx 4.3$ GB per layer — utterly impossible on any edge device, and a strain even on a datacenter GPU. Linear attention's largest intermediate is the $d \times d$ state $S$, which is $8 \times 64 \times 64 \times 2$ bytes $\approx 64$ KB per layer — five orders of magnitude smaller, and it does **not grow with $n$ at all**. Push $n$ to a million and that state is still 64 KB. That flat-in-$n$ memory is why linear attention is the right tool when the sequence is genuinely enormous and the device is genuinely tiny; it is also why it underpins streaming/recurrent inference, where the state *is* the model's memory of everything it has seen.

**The causal-masking subtlety.** For autoregressive generation each token may attend only to earlier tokens, which seems to break the "compute $S$ once" trick — token $i$ needs $S_i = \sum_{j \le i} \phi(k_j) v_j^\top$, a *different* partial sum for each position. But these partial sums form a running prefix: $S_i = S_{i-1} + \phi(k_i) v_i^\top$. So causal linear attention is literally a recurrent neural network — a fixed-size state updated one token at a time, $O(1)$ memory and $O(1)$ compute *per generated token*, $O(n)$ overall. This is the "transformers are RNNs" punchline, and it is genuinely attractive for on-device streaming (speech, live captioning) where you generate token by token and a fixed-size state means constant memory regardless of how long the stream runs — no growing KV cache. The cost is the same approximation: the recurrent state is a lossy summary, so very long-range exact recall suffers.

**What linear attention gives up.** This is the honest part. The softmax has a *peaky* quality — it can put almost all the weight on one or two keys, which matters when a token needs to attend sharply to exactly one other token (think induction heads in language, or a salient object in an image). Linear feature maps are *smooth*; they struggle to reproduce a near-one-hot attention distribution. In practice linear attention is excellent when attention is diffuse (broad context mixing) and noticeably worse when it needs to be sharp. It also changes the causal-masking story: with a causal mask the running state $S$ becomes a *recurrence* (this is why the paper is called "transformers are RNNs"), which is great for streaming inference but means the parallel-prefix trick costs more care. For most edge *vision* workloads — where you want global context mixing rather than razor-sharp single-token lookups — linear attention is a strong fit. For language with hard retrieval, be careful.

There is a whole family beyond pure linear attention, and they trade off differently:

- **Windowed / local attention (Swin, Longformer).** Restrict each token to attend only to a window of $w$ neighbors. Cost drops to $O(n w d)$ — linear in $n$ for fixed window. Swin Transformer (Liu et al., 2021) makes this work for vision by *shifting* the windows between layers so information leaks across window boundaries, recovering a global receptive field over a few layers. It is *exact* within each window and gives up only long-range links in a single layer. This is my default for high-resolution vision: it maps cleanly to hardware (every window is an independent small attention) and it keeps the convolution-like locality that images actually have. Longformer applies the same idea to language with a sliding window plus a handful of *global* tokens (the classification token, question tokens) that attend everywhere — a cheap way to keep a few long-range links while paying linear cost for the rest, which is exactly the pattern you want when most of a long document is locally coherent but a few anchors need global reach.
- **Low-rank attention (Linformer).** Project the keys and values down the *sequence* dimension from $n$ to a small $k$ before attention: $Q (E K)^\top$ where $E$ is $k \times n$. Now scores are $n \times k$, cost $O(nkd)$. It assumes the attention matrix is approximately low rank (often true). It gives up the ability to attend to fine sequence detail beyond rank $k$, and the projection $E$ is learned for a fixed $n$, which is awkward for variable-length inputs.

Why does the shifted-window scheme matter enough to highlight? Plain windowed attention has a fatal flaw for vision: a token at a window boundary can never attend to its neighbor one pixel across the boundary, so information is trapped in each window forever. Swin's fix is almost trivially elegant — in alternating layers, shift every window by half its size, so the boundaries fall in different places and a token that was at the edge of one window sits in the middle of another next layer. After two layers every token has an indirect path to every other token within a couple of window-widths, and after the full hierarchy the receptive field is global. You pay strictly linear cost and get an effectively global model, with the bonus that the patch-merging between stages builds the multi-resolution pyramid that detection and segmentation heads expect. The practical reason I default to windowed for vision over linear: it is *exact within the window*, so it never blurs the sharp local attention that fine textures need, and it is trivially parallel (each window is an independent small attention, perfect for batching on an NPU). Linear attention's smoothness is a liability exactly where vision wants sharpness.

We will compare all four head-to-head in the results section. First, the variant that gives up *nothing* mathematically.

## 3. FlashAttention: same math, far less memory traffic

Here is the part that surprises people. FlashAttention (Dao et al., 2022) computes *bit-for-bit the same attention output* as the vanilla implementation — same FLOPs, exact softmax, no approximation. And it is several times faster and uses an order of magnitude less memory. How can the same arithmetic be faster? Because on modern accelerators, attention is not compute-bound. It is **memory-bound**. The bottleneck is moving the $n \times n$ matrix between fast on-chip SRAM and slow off-chip HBM (high-bandwidth memory), not the multiply-adds. This is the roofline model talking, and it is the single most important systems insight in this post — go read [the roofline model post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) if "arithmetic intensity" is not yet second nature.

![Side by side comparison of standard attention writing the full score matrix to slow memory versus FlashAttention tiling the work and running an online softmax in fast on-chip memory](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-3.png)

Walk through the standard kernel's memory traffic. Compute $S = QK^\top$ → **write** $n^2$ values to HBM. Read them back, compute softmax → **write** $n^2$ values back. Read them back again, multiply by $V$ → write the $n \times d$ output. The big matrix crosses the slow memory bus *three times*. Arithmetic intensity — FLOPs per byte moved — is low, so the GPU's matrix units sit idle waiting on memory. You are paying for a Ferrari and driving it in a parking lot.

FlashAttention's fix is **tiling plus an online softmax** so the $n \times n$ matrix never has to exist in HBM. Split $Q$, $K$, $V$ into blocks that fit in SRAM. For each block of queries, stream over the blocks of keys/values, computing partial attention and *accumulating* the result, keeping only the running statistics you need. The trick that makes this exact is the **online softmax recurrence**.

The problem with computing softmax incrementally is the normalization. $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$, and for numerical stability you subtract the row max: $e^{x_i - m} / \sum_j e^{x_j - m}$ with $m = \max_j x_j$. But you do not know the global max until you have seen every key. The online trick: maintain a running max $m$, a running denominator $\ell$, and a running output accumulator $o$, and *correct* them as new blocks arrive. When a new block has local max $m'$ and contributes local sum $\ell'$ and local output $o'$, the new global max is $m^{\text{new}} = \max(m, m')$ and you rescale:

$$
\ell^{\text{new}} = e^{m - m^{\text{new}}}\,\ell + e^{m' - m^{\text{new}}}\,\ell', \qquad
o^{\text{new}} = e^{m - m^{\text{new}}}\,o + e^{m' - m^{\text{new}}}\,o'
$$

The factors $e^{m - m^{\text{new}}}$ retroactively rescale everything you have accumulated so far to the new max. After the last block, divide $o$ by $\ell$ and you have the exact softmax-weighted output — the *same* number the naive method would produce, computed without ever materializing a full row of scores in HBM. The big matrix lives and dies inside SRAM, one tile at a time.

Let me make the recurrence concrete with a tiny trace so the rescaling is not mysterious. Suppose a query's scores against the keys, processed in two blocks, are $[2, 4]$ then $[6, 1]$. The true softmax weights are over all four: max is 6, $e^{2-6}+e^{4-6}+e^{6-6}+e^{1-6} = 0.018 + 0.135 + 1 + 0.0067 \approx 1.16$. Now do it online. Block 1: local max $m = 4$, denominator $\ell = e^{2-4} + e^{4-4} = 0.135 + 1 = 1.135$, accumulator $o$ built with those weights. Block 2 arrives with local max $m' = 6$. The new global max is $m^{\text{new}} = 6$. We rescale the old denominator by $e^{m - m^{\text{new}}} = e^{4-6} = 0.135$: $0.135 \times 1.135 = 0.153$. We add the new block's contribution $e^{6-6} + e^{1-6} = 1 + 0.0067 = 1.0067$, also at the new scale (its local max already is the new max so its factor is 1): total $\ell = 0.153 + 1.0067 = 1.16$ — exactly the true denominator. The accumulator $o$ gets the same $0.135$ correction so the already-accumulated values are retroactively renormalized to the new max. After the last block, $o / \ell$ is the exact output. No row of scores ever lived in full; the state is three small running numbers per query.

The payoff is dramatic and it is a *memory-traffic* payoff: HBM accesses drop from $O(n^2)$ to $O(n)$ (you read $Q, K, V$ and write the output, each $O(nd)$, plus modest extra). Concretely, the arithmetic intensity — useful FLOPs per byte moved — climbs from roughly $O(1)$ (the naive kernel moves a byte for nearly every multiply) to $O(d)$ or better, which on the roofline pushes attention from the bandwidth-limited slope onto the compute-limited ceiling where the tensor cores can actually run flat-out. Dao et al. report 2–4× wall-clock speedups on long sequences and the ability to fit context lengths that simply OOM'd before; FlashAttention-2 (2023) roughly doubled that again by improving the work partitioning across the GPU's warps and reducing non-matmul FLOPs. On the edge this matters even more, because the gap between on-chip SRAM bandwidth and off-chip DRAM bandwidth is *wider* on a phone or a Jetson than on a datacenter GPU — the memory-bound penalty is harsher, so removing it helps more. The catch is that FlashAttention is a hand-written fused kernel: it exists for CUDA, for Apple's Metal Performance Shaders (so Core ML and MLX get a version), for some mobile GPUs, but *not* universally — and if your target runtime lacks a fused attention kernel, you fall back to the materialize-the-matrix path and lose the win. Checking kernel availability on the actual target is step zero, not an afterthought.

The cleanest way to get FlashAttention without writing CUDA is through PyTorch's fused attention. Since PyTorch 2.0, `torch.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to a FlashAttention backend automatically when the shapes and dtype allow it:

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# q, k, v: (batch, heads, seq_len, head_dim)
B, H, N, D = 1, 12, 4096, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

# Force the FlashAttention backend so we know which kernel ran.
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

print(out.shape)  # torch.Size([1, 12, 4096, 64])
```

That single call replaces the explicit `softmax(q @ k.transpose(-2, -1) / scale) @ v` and, on supported hardware, runs the tiled exact kernel. The flags that matter: FlashAttention wants fp16 or bf16 (it is built for the tensor-core path), the head dimension must be supported (≤256), and you should pass `is_causal=True` rather than a materialized mask when you want causal masking, because an explicit boolean mask forces a fallback to the math backend that *does* materialize the matrix — quietly throwing away the win. Always confirm which backend ran; I have been burned by a stray attention mask silently disabling flash.

## 4. Measuring the FlashAttention win honestly

A speedup you have not measured is a speedup you do not have. Here is a self-contained micro-benchmark that pits naive attention against SDPA's flash backend at a long sequence, measuring both peak memory and latency the way you would on a real device — with warm-up, with synchronization, batch size 1.

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

device = "cuda"
B, H, N, D = 1, 12, 8192, 64
dtype = torch.float16

def make():
    g = lambda: torch.randn(B, H, N, D, device=device, dtype=dtype)
    return g(), g(), g()

def naive(q, k, v):
    scale = 1.0 / (D ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale   # materializes B,H,N,N
    probs = scores.softmax(dim=-1)
    return probs @ v

def flash(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

def bench(fn, iters=50):
    q, k, v = make()
    for _ in range(10):          # warm-up: trigger autotune, fill caches
        fn(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return ms, peak_mb

for name, fn in [("naive", naive), ("flash", flash)]:
    try:
        ms, mb = bench(fn)
        print(f"{name:6s}  {ms:7.2f} ms   peak {mb:8.1f} MB")
    except RuntimeError as e:
        print(f"{name:6s}  OOM / error: {str(e)[:60]}")
```

#### Worked example: vanilla versus FlashAttention at a long sequence

Running the benchmark above on an NVIDIA-class accelerator with $N = 8192$ tokens, 12 heads, head dim 64, fp16, batch 1, the numbers come out roughly like this (your exact figures depend on the chip; these are representative and the *ratios* are the point):

| Implementation | Latency (ms, batch 1) | Peak memory | Notes |
| --- | --- | --- | --- |
| Naive `softmax(QKᵀ)V` | ~6.1 ms | ~3.2 GB | materializes 12×8192² fp16 scores |
| SDPA FlashAttention | ~1.9 ms | ~0.35 GB | no full score matrix in HBM |

That is roughly a **3× latency** improvement and a **~9× memory** reduction, for *identical output values* (compare them with `torch.allclose` at a loose tolerance and they match). The memory number is the one that decides whether a long-context or high-resolution model *runs at all* on a memory-constrained device — and on a Jetson Orin Nano with 8 GB shared between CPU and GPU, or a phone with a few hundred MB of practical headroom, "runs at all" is most of the battle. The latency number is real too, and it is purely a memory-traffic win: the FLOPs are the same in both rows. If you measure FLOPs you would conclude they are equal; the roofline tells you why one is 3× faster. Honest-measurement caveats: warm up first (the first call triggers kernel autotuning and looks artificially slow), always `synchronize()` before reading the clock (CUDA is async — without sync you are timing the launch, not the work), and report batch-1 latency because that is the on-device reality, not the throughput-optimized batch-256 number that looks better in a slide.

The naive row may simply OOM at this sequence length on a smaller device — which is the cleanest possible demonstration of the point. When it OOMs, FlashAttention is not "3× faster," it is the difference between shipping and not shipping.

## 5. Efficient vision transformers: let convolutions do the cheap work

Attention-level tricks are necessary but not sufficient for vision on the edge. The deeper problem with a *pure* ViT on a small device is structural: it throws away the strongest prior we have about images. Convolutions bake in **locality** (nearby pixels relate) and **translation equivariance** (a cat is a cat wherever it appears) for free, with cheap, hardware-friendly ops. A pure ViT has to *learn* those priors from data, which is why ViTs are famously data-hungry and why, at the small sizes the edge demands, they underperform a well-tuned CNN of the same budget. The winning move is a **hybrid**: use convolutions for the early, high-resolution, many-token stages where attention would be ruinously expensive and where locality is exactly what you want, and use transformer blocks only at the later, low-resolution stages where global mixing earns its cost.

### MobileViT: a conv-transformer block

MobileViT (Mehta & Rastegari, 2022) is the cleanest expression of this idea, and it is the one I reach for first. Its block (Figure 5) does three things: a cheap local representation with standard convolutions, a *global* representation with a small transformer applied over unfolded patches, and a fusion that combines them so both inductive biases survive.

![Dataflow graph of a MobileViT block showing a local convolution branch and a global transformer branch that merge in a fusion step](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-5.png)

Concretely: the input feature map goes through a 3×3 conv (local features) then a 1×1 conv to a transformer dimension. Then the map is **unfolded** into non-overlapping patches and the transformer attends *across patches at the same position* — so token count is the number of patch positions, far smaller than per-pixel, and attention cost stays bounded. The result is **folded** back into a spatial map, projected with a 1×1 conv, and **concatenated with the original local features** before a final fusion conv. That concat is the load-bearing detail: it is why MobileViT keeps the convolutional inductive bias instead of replacing it. The transformer adds long-range mixing on top of, not instead of, the cheap local structure. MobileViT-S hits ~78.4% ImageNet top-1 with ~5.6M parameters — beating MobileNetV3 at a similar parameter count, and beating a much larger pure ViT that needs heavy data augmentation and pretraining to compete.

There is a deeper reason hybrids win, and it is worth stating plainly because it guides where you place transformer blocks. A pure ViT applies global attention at *every* layer, including the early ones where the feature map is large and the features are low-level (edges, textures) that have no business being mixed globally — you are paying the quadratic to let a corner pixel attend to the opposite corner before either has formed a useful representation. The hybrid recognizes that early layers want *local* processing (cheap convs, large feature map, many implicit tokens) and only *late* layers, where the resolution has been downsampled and features are semantic, benefit from global mixing (expensive attention, but now over few tokens). This is the same lesson as the convolution-stem ViTs (Xiao et al., 2021, *Early Convolutions Help Transformers See Better*): replacing the ViT's naive patchify-stem with a few convolutions stabilizes training and improves accuracy, because convolutions are simply the right tool for the high-resolution early stages. Put the attention where global context is worth its cost — late, low-resolution — and let convolutions own the rest.

### EfficientViT: making the attention itself cheap

MobileViT keeps standard softmax attention but feeds it few tokens. EfficientViT (Cai et al., 2023) attacks the attention op directly with two ideas. First, **ReLU linear attention** — the $\phi(x)$ kernel trick from Section 2 — so the attention is $O(n)$, which matters because EfficientViT targets high-resolution dense tasks (segmentation, super-resolution) where even bounded patch counts get large. Second, **cascaded group attention**: split the heads into groups and feed each successive group the *output* of the previous one, which adds depth-like feature diversity without adding heads, and cuts the redundant computation that plain multi-head attention wastes. The combination gives EfficientViT genuinely high throughput on edge GPUs — it is one of the few ViTs that is *faster* than a comparable-accuracy CNN on a Jetson, not just smaller.

Cascaded group attention is a subtle but important fix for a known waste in multi-head attention: the heads are *redundant*. Studies of trained transformers find many heads learn near-identical attention patterns, so the full $h$-way parallel attention spends compute on duplicates. Cascading forces diversity structurally — each head group sees a different input (the running sum of prior groups' outputs), so it *has* to learn something the others did not, and you get the representational benefit of many heads at the compute of fewer. The diagnostic lesson generalizes beyond EfficientViT: if you profile a transformer and find attention-head redundancy, that redundancy is latency you are paying for nothing, and either head pruning (see [pruning LLMs and transformers](/blog/machine-learning/edge-ai/pruning-llms-and-transformers)) or a cascaded design recovers it.

A note on EfficientViT's choice of linear attention over windowed: it is deliberate. For *dense* prediction at very high resolution (1024×1024 segmentation), windowed attention's fixed window means the receptive field grows only slowly with depth, so you need many layers to see globally — whereas linear attention is global in one layer at $O(n)$ cost. EfficientViT bets that global-but-approximate beats local-but-exact for dense tasks, and the benchmarks back it. For *classification* at modest resolution the bet is less clear, which is part of why MobileViT (softmax + few tokens) and EfficientViT (linear) coexist rather than one dominating — they target different points on the resolution axis.

### Token merging: spend tokens where they matter

Even with a hybrid, the transformer stages still pay per token. **Token Merging (ToMe)** (Bolya et al., 2023) is a beautifully cheap, *training-free* way to cut the sequence: between attention blocks, find the $r$ most similar token pairs with a fast bipartite soft-matching and average each pair into one token. The sequence shrinks by $r$ tokens per block, so attention gets progressively cheaper as you go deeper, and because nearby image patches are genuinely redundant, accuracy barely moves.

![Side by side comparison of a transformer keeping every token versus token merging averaging similar token pairs to shrink the sequence](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-6.png)

ToMe gives roughly 2× throughput on a ViT for a 0.2–0.4 point accuracy drop, with *no retraining* — you can bolt it onto an existing model at inference time. That "free" quality is rare and makes it a great first thing to try when an existing ViT is just slightly too slow. Pooling/strided downsampling of tokens (used in PVT, MViT, and Swin's patch-merging) is the trained cousin: instead of merging by similarity, you periodically halve the spatial resolution of the token grid, which both cuts tokens and builds the multi-scale pyramid that dense vision tasks want. The common thread across MobileViT, EfficientViT, and token reduction is one sentence: **on the edge, the cheapest token is the one you never compute attention over.**

#### Worked example: token merging on a ViT-B classifier

Put numbers on the "free" win. A ViT-B/16 at 224×224 has $14 \times 14 + 1 = 197$ tokens through all 12 layers, so each attention block is $197^2 \approx 38{,}800$ score entries per head. Apply ToMe with $r = 16$ merges per block: after block 1 the sequence is 181 tokens, after block 2 it is 165, and so on, falling to about $197 - 11 \times 16 = 21$ tokens by the final block (you stop merging before it gets too aggressive). Because attention cost is quadratic in the *current* token count, the deep blocks — which used to cost the same as the shallow ones — now cost a fraction: the last block's attention is $21^2 / 197^2 \approx 1\%$ of its original cost. Summed over the network, throughput roughly doubles and ImageNet top-1 drops from, say, 81.8% to about 81.4–81.6% — a few tenths of a point for a 2× speedup, with not one weight retrained. On a Jetson that can turn a 14 ms classifier into a 7–8 ms one over a lunch break. The one caveat: ToMe changes which tokens exist, so if a *downstream* head needs a fixed grid (dense segmentation), you either un-merge at the end or use a grid-preserving reduction instead — for plain classification, where you pool to a single vector anyway, it is pure upside.

### Practical: load a hybrid ViT from timm

You do not implement these from scratch. The `timm` library has MobileViT, EfficientViT, and dozens of hybrids pretrained:

```python
import timm
import torch

# Pretrained MobileViT-S; timm name encodes the variant.
model = timm.create_model("mobilevit_s", pretrained=True)
model.eval()

# A MobileNetV3-Large baseline to compare against.
cnn = timm.create_model("mobilenetv3_large_100", pretrained=True)
cnn.eval()

x = torch.randn(1, 3, 256, 256)  # MobileViT trains at 256
with torch.inference_mode():
    logits = model(x)
print(logits.shape)  # torch.Size([1, 1000])

# Count params to sanity-check the budget.
def millions(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
print(f"mobilevit_s  {millions(model):.1f} M params")
print(f"mnv3_large   {millions(cnn):.1f} M params")
```

And a minimal linear-attention forward that makes the associativity reorder explicit — this is the code form of Figure 2, so you can see that the $O(n)$ path is a one-line change in *where you multiply*:

```python
import torch
import torch.nn.functional as F

def linear_attention(q, k, v, eps=1e-6):
    """O(n) attention via the kernel trick.
    q, k, v: (batch, heads, seq, dim). phi = elu + 1 keeps features positive."""
    phi = lambda x: F.elu(x) + 1.0
    q, k = phi(q), phi(k)                      # feature maps
    # Build the d x d state ONCE: sum_j phi(k_j) v_j^T
    kv = torch.einsum("bhnd,bhne->bhde", k, v)        # (b,h,d,d)  <- the reorder
    z = k.sum(dim=2)                                  # (b,h,d)    normalizer keys
    num = torch.einsum("bhnd,bhde->bhne", q, kv)      # (b,h,n,d)
    den = torch.einsum("bhnd,bhd->bhn", q, z).unsqueeze(-1) + eps
    return num / den                                   # no n x n matrix anywhere

B, Hh, N, Dd = 1, 8, 16384, 64
q = torch.randn(B, Hh, N, Dd)
k = torch.randn(B, Hh, N, Dd)
v = torch.randn(B, Hh, N, Dd)
out = linear_attention(q, k, v)
print(out.shape)  # torch.Size([1, 8, 16384, 64]) — at N=16384 vanilla would need a 16384^2 matrix
```

The two `einsum` calls are the whole trick: `bhnd,bhne->bhde` contracts over the sequence index `n` *first*, producing the compact $d \times d$ state, so the sequence length never appears squared. At $N = 16384$ a vanilla score matrix would be 268 million entries per head; here the largest intermediate is $64 \times 64$.

## 6. The attention-variants comparison, side by side

You now have four ways to make attention affordable, and they are not interchangeable. Figure 4 lays out the trade space, and the table after it adds the numbers and the "when."

![Comparison matrix of vanilla linear windowed and FlashAttention across complexity exactness and what each one gives up](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-4.png)

| Variant | Compute | Memory | Exact? | Gives up | Reach for it when |
| --- | --- | --- | --- | --- | --- |
| **Vanilla** | $O(n^2 d)$ | $O(n^2)$ | exact | nothing | short sequences; baseline correctness |
| **FlashAttention** | $O(n^2 d)$ | $O(n)$ | **exact** | needs the fused kernel | long sequences, memory-bound; you have the kernel |
| **Linear (kernel)** | $O(n d^2)$ | $O(n d)$ | approximate | sharp/peaky attention | diffuse global mixing; very long $n$; streaming |
| **Windowed (Swin/Longformer)** | $O(n w d)$ | $O(n w)$ | exact in window | single-layer long-range links | high-res vision; locality is real |
| **Low-rank (Linformer)** | $O(n k d)$ | $O(n k)$ | approximate | fine detail beyond rank $k$; fixed $n$ | low-rank attention; fixed-length inputs |

The decisive distinction is the **Exact?** column. FlashAttention is the only one that costs you *nothing* in accuracy — it is a pure systems optimization, and so it should be your *first* move, always, before you reach for any approximation. Only when FlashAttention is not enough (the kernel is unavailable on your target, or even $O(n)$ memory at the exact FLOP count is too slow) do you trade exactness: windowed when locality is genuinely real (vision, local language structure), linear when attention is diffuse and the sequence is enormous, low-rank when the inputs are fixed-length and the attention really is low-rank. A practical edge stack often *layers* these: a windowed-attention vision transformer whose windows are each computed with a FlashAttention kernel, then token-merged to shrink the grid. They compose because they attack different axes — windowing reduces *which* tokens attend, flash reduces *how* the attention is moved through memory, merging reduces *how many* tokens exist.

## 7. Case studies and real numbers

The literature has done the careful measurement; here are the results I trust, with sources, so you can calibrate expectations before you build.

**MobileViT vs MobileNetV3 vs ViT (Mehta & Rastegari, 2022).** On ImageNet-1k, MobileViT-S reaches ~78.4% top-1 with ~5.6M parameters. MobileViT-XS hits ~74.8% with ~2.3M. The comparison that matters: at a comparable parameter budget MobileViT *beats* MobileNetV3, and it does so without the giant pretraining datasets a pure DeiT/ViT needs to be competitive at small sizes. The honest caveat the paper itself flags: MobileViT's *latency* on a phone can be worse than its FLOP count suggests, because the unfold/fold reshaping and the attention ops are less optimized in mobile inference engines than plain convolutions — a vivid reminder that FLOPs are not latency (the theme of the [roofline](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) and the efficient-architecture posts). Measure on the target.

#### Worked example: MobileViT vs MobileNetV3 vs ViT-Small on a Jetson Orin Nano

Put the three on one named target — a Jetson Orin Nano, batch 1, 256×256 input, fp16, latencies p50 from warmed runs. The figures below are representative of what these models do on this class of device; treat them as order-of-magnitude and measure your own:

| Model | Top-1 | Params | GFLOPs | Latency p50 (Orin Nano, fp16) | Verdict |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | ~75.2% | 5.4 M | 0.22 | ~3.5 ms | fastest; CNN baseline to beat |
| MobileViT-S | ~78.4% | 5.6 M | 2.0 | ~9 ms | +3.2 pts acc, ~2.5× slower |
| EfficientViT-B1 | ~79.4% | 9.1 M | 0.52 | ~5 ms | best accuracy/latency point |
| ViT-Small/16 | ~80.2% | 22 M | 4.6 | ~16 ms | +5 pts but 4× params, 4–5× latency |

![Comparison matrix of MobileNetV3 MobileViT EfficientViT and ViT-Small across top-1 accuracy parameters and compute](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-7.png)

Read that table the way an edge engineer must: the *Pareto frontier* runs MobileNetV3 → EfficientViT-B1, with MobileViT-S a reasonable interior point and ViT-Small mostly dominated for on-device use (it buys ~2 points over EfficientViT for ~2.4× the parameters and ~3× the latency). The transformer is "worth it" here only because the hybrids closed the latency gap; a *pure* ViT-Small is not worth it on this device for classification. The decision flips for a task where global context dominates — fine-grained recognition, document layout, dense prediction — where attention's long-range mixing earns its cost and the hybrids pull further ahead of the CNN.

**Swin Transformer (Liu et al., 2021).** Swin made windowed attention the workhorse of high-resolution dense vision: shifted local windows give linear complexity in image size while a hierarchy of patch-merging stages builds the multi-scale pyramid that detection and segmentation need. Swin-T matches or beats a ResNet-50-class backbone on detection/segmentation at comparable cost, and the *windowed* attention is what makes 800×1333 detection inputs tractable — vanilla global attention at that resolution is simply not an option. This is the paper to cite when someone proposes full global attention on a high-res image; point them at windows.

**FlashAttention in production (Dao et al., 2022; FlashAttention-2, 2023).** The original paper reported 2–4× attention speedups and made it routine to train and serve at long context (the memory went from quadratic to linear, so the achievable context length jumped severalfold). It is now the default kernel behind essentially every serving stack and behind PyTorch SDPA. The lesson for the edge is the transferable one: a large fraction of "this transformer is too slow" is memory traffic, not arithmetic, and the fix is to stop moving the big matrix — not to do less math. For the LLM-serving angle on the same insight — the KV cache as the memory-bound bottleneck of decoding — see [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

**Token Merging (Bolya et al., 2023).** ToMe reports ~2× throughput on ViT-L/ViT-H for a fraction of a point of accuracy, training-free, and even *better* with a light fine-tune. It is the cheapest win in this whole post because it requires no new training and no kernel — just a similarity match between blocks — which makes it the ideal "we're 30% over budget and ship in a week" lever.

**On-device language models (Gemma, Phi, Llama on phones).** The same attention economics drive the small language models now running on phones. The dominant cost in autoregressive decoding is not the attention compute but the **KV cache**: every generated token must read the keys and values of all prior tokens, so memory traffic grows with context length and decoding becomes memory-bound — the language-model mirror of the vision quadratic. The fixes rhyme with this post: FlashAttention's tiling for the prefill, grouped-query and multi-query attention (fewer KV heads, less cache to move), and sliding-window/local attention (Mistral, Gemma) to bound the cache. On-device 2–4B models lean on all of these plus aggressive int4 weight quantization. The architectural lever and the quantization lever compose: see [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) for the decoding-side mechanics and [multi-head latent attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) for the most aggressive KV-compression scheme. The transferable lesson for any edge transformer: identify whether your bottleneck is prefill (compute, fix with flash/efficient attention) or decode (memory traffic, fix with KV-cache shrinking), because they want different tools.

## 8. When a transformer is worth it on the edge — and when a CNN wins

After all this, the most useful thing I can give you is a decision, not a menu. Figure 8 is the tree I actually use.

![Decision tree for choosing edge attention by whether a fused kernel exists whether the sequence is long and whether the task has local structure](/imgs/blogs/efficient-attention-and-vision-transformers-for-edge-8.png)

Start from the bottleneck, not from the model name:

- **If you have a fused attention kernel on your target, keep exact attention (FlashAttention/SDPA).** Never approximate before you have removed the memory-traffic waste. This is free accuracy.
- **If the sequence is long and you are memory-bound even with flash, approximate:** windowed when locality is real (almost always true for vision), linear when attention is diffuse and $n$ is huge, low-rank for fixed-length inputs.
- **If it is a vision task with strong local structure, prefer a hybrid (MobileViT/EfficientViT) over a pure ViT,** and consider whether a plain CNN already meets the target — because frequently it does.

And the honest counter-position: **a CNN often wins on the edge, and you should be glad when it does.** For straightforward classification or detection at modest resolution, a well-tuned [MobileNet-family](/blog/machine-learning/edge-ai/the-mobilenet-family) CNN is smaller, faster, better-supported by mobile inference engines, easier to quantize, and gives up only a couple of accuracy points — points you may not need. The support gap is the part teams underestimate: convolutions, depthwise convolutions, and pooling have been the bread and butter of mobile inference engines for years, so they are heavily optimized, reliably accelerated on every NPU, and quantize cleanly. Transformer ops — the attention pattern, the unfold/fold reshapes, LayerNorm, GELU — are newer, less uniformly supported, and more likely to trigger a CPU fallback or a precision-sensitive quantization edge case. A CNN that is 2 points less accurate but runs entirely on the NPU at half the latency and quantizes to int8 with no drama is, for most products, the better engineering decision. Accuracy on a slide is not the metric; accuracy *at the latency, power, and reliability the product needs* is, and the CNN frequently owns that point on the frontier. Reach for a transformer on the edge when **(a)** the task genuinely needs long-range or global context that convolutions capture only weakly (fine-grained recognition, document/scene understanding, multi-object reasoning, anything where "what's across the image" matters), **(b)** you can afford the hybrid's latency on the target *after* measuring (not after estimating from FLOPs), and **(c)** the accuracy gain clears your product's bar. If two of those three are shaky, ship the CNN. The same accessibility-first instinct shows up on the language side: a well-chosen small model often beats a shrunk-down big one, which is the argument of [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design).

### Composing with quantization — and the ViT int8 trap

Efficient architecture is the first lever; you still apply the others. Once you have a hybrid ViT that meets the latency budget in fp16, the obvious next move is int8 quantization to halve the size and lean on the NPU's int8 path. **But transformers do not quantize like CNNs, and ViTs have a specific landmine: activation outliers.** The inputs to attention — and especially the post-LayerNorm and GELU activations — develop a few channels with values an order of magnitude larger than the rest. A naive per-tensor int8 quantizer sets its scale to cover those outliers, which crushes the resolution of the *normal* values into a handful of int8 levels, and accuracy collapses. This is exactly the activation-outlier problem that motivated SmoothQuant and per-channel/per-token schemes; the mechanics and the fix transfer directly from the LLM world, so read [LLM quantization: activations, SmoothQuant, and the KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) before you int8 a ViT.

The practical recipe that works: keep the LayerNorm and softmax in higher precision (they are cheap and outlier-sensitive), use *per-channel* weight quantization and *per-token* (or smoothed) activation quantization, and calibrate on enough representative data to actually see the outlier distribution. Here is a PyTorch sketch of static int8 PTQ on the convolutional/linear backbone with the sensitive ops left in float:

```python
import torch
from torch.ao.quantization import quantize_fx, get_default_qconfig_mapping

model = timm.create_model("mobilevit_s", pretrained=True).eval()

# Per-channel int8 weights + per-token-ish activation observers (x86 default
# uses histogram observers that respect the activation range).
qconfig_mapping = get_default_qconfig_mapping("x86")

example = torch.randn(1, 3, 256, 256)
prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example)

# Calibrate on REAL data so the observers see the activation outliers.
with torch.inference_mode():
    for images in calibration_loader:   # a few hundred representative images
        prepared(images)

int8_model = quantize_fx.convert_fx(prepared)
```

If int8 still loses too much, the escalation ladder is: smooth the activations first (shift the outlier scale from activations into weights, à la SmoothQuant), then mixed precision (keep attention in fp16 and quantize only the feed-forward and conv layers — see [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis)), then quantization-aware training as the last resort. The general PTQ-vs-QAT logic from [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) holds: do not pay for QAT until PTQ provably misses the target.

**Stress-testing the decision.** What breaks the "use a hybrid ViT" recommendation? Several real things, and you should know them before you commit. *The op falls back to CPU.* If the NPU does not support the unfold/fold reshapes or the specific attention pattern, the framework silently runs them on the CPU, and the round trips between NPU and CPU dominate latency — a model that looks great in FLOPs runs 5× slower than the CNN. Profile the actual op placement, not the graph. *The calibration set is tiny.* ViT activation outliers are distribution-dependent; calibrate on 50 images and you will miss outlier channels that appear in production, and the int8 model degrades on real inputs. Use hundreds, drawn from the true distribution. *The sequence is short.* At small token counts ($n \lesssim 256$) the quadratic is not the bottleneck — the feed-forward layers and the memory layout are — and FlashAttention's win shrinks; do not over-optimize attention when it is not the cost. *Memory-bound, not compute-bound.* If your profile says the model is memory-bound (low arithmetic intensity), more FLOP-efficient attention will not help — you need to reduce *data movement* (flash, fewer tokens, fused ops), which is why the roofline read comes first.

*What about int4 for a ViT?* The escalation does not stop at int8. Sub-8-bit (int4 and below, covered in [sub-8-bit networks](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks)) is where vision transformers get genuinely fragile, and for a specific reason: the activation-outlier problem that int8 merely strains, int4 amplifies, because you now have only 16 levels to cover both the outliers and the dense bulk of normal values. Weight-only int4 (the GPTQ/AWQ style that dominates LLMs) survives better than activation int4 on ViTs, but even weight-only int4 on attention projections can drop a couple of accuracy points unless you protect the most sensitive layers. The defensible recipe at int4 is mixed: int4 weights on the FFN and projection layers, int8 (or fp16) on the attention scores and the first/last layers, and a short QAT pass to recover the loss. If you are tempted to int4 a ViT to hit a flash-memory budget, budget for the accuracy recovery work too — it is rarely free the way int8 often is.

*When does linear attention itself break?* It breaks when the task needs *sharp* attention — a single token that must dominate a row. Induction and retrieval heads in language are the classic case: copying a specific earlier token requires a near-one-hot attention row that a smooth kernel feature map cannot represent well, so linear-attention language models lag on exact recall and long-context retrieval benchmarks even when they match on perplexity. In vision the failure is subtler but real: very fine-grained discrimination (telling near-identical bird species apart) can need a sharp focus on a discriminative patch, and linear attention's smoothing blurs it. The honest test is to swap linear attention in and measure on the *hardest* slice of your eval set, not the average — the average can look fine while the tail collapses. If the tail matters (safety-critical recognition, retrieval), keep exact attention (flash) and find your savings elsewhere.

One more practical guard worth its own snippet: silently losing the flash backend is the most common way teams *think* they shipped FlashAttention and did not. PyTorch will quietly fall back to the math backend if anything about the call is unsupported — a materialized mask, an odd head dimension, an unsupported dtype — and you get correct results at quadratic memory. Assert the backend rather than hope:

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

q = torch.randn(1, 12, 4096, 64, device="cuda", dtype=torch.float16)
k, v = q.clone(), q.clone()

# Restrict to ONLY the flash backend. If it cannot run, this RAISES
# instead of silently falling back to the quadratic-memory math kernel.
try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print("flash backend ran; peak MB:",
          torch.cuda.max_memory_allocated() / 1e6)
except RuntimeError as e:
    print("flash unavailable for these shapes/dtype:", str(e)[:80])
    # Now you KNOW, and can fix the shape/dtype or accept the fallback.
```

Wrapping production attention in an explicit-backend context (or at least logging which backend `torch.backends.cuda.sdp_kernel` selected) turns a silent quadratic regression into a loud failure you can fix before it ships — exactly the kind of guard that would have saved that first crashed demo.

## 9. Profiling: knowing which wall you are hitting

Every recommendation above branches on one question — *are you compute-bound or memory-bound?* — and the only honest way to answer it is to measure on the target. Here is the profiling flow I run before touching a single attention kernel, because optimizing the wrong wall is the most common way to waste a sprint.

The first cut is a roofline placement: compute the model's arithmetic intensity (FLOPs per byte of memory traffic) and compare it to the device's ridge point (peak FLOPs ÷ peak bandwidth). If your intensity is below the ridge, you are bandwidth-limited and FLOP-cutting tricks will not help; you need to move less data (FlashAttention, fewer tokens, operator fusion). If above, you are compute-limited and reducing FLOPs (linear/windowed attention, smaller heads, pruning) pays off. Attention's score-matrix step is almost always below the ridge — that is the whole reason FlashAttention exists — but the FFN and the conv stages can be either, so measure per-stage, not for the whole model.

The second cut is an operator-level trace to find where the time actually goes, and critically, *where ops execute*. The PyTorch profiler gives you both:

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

model = timm.create_model("mobilevit_s", pretrained=True).eval().cuda()
x = torch.randn(1, 3, 256, 256, device="cuda")

# Warm up so we trace steady-state, not first-call autotune.
with torch.inference_mode():
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("forward"):
            model(x)
        torch.cuda.synchronize()

# Sort by total CUDA time to see the real hot ops.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

Read the table for three things. **One:** which op family dominates — if it is `aten::scaled_dot_product_attention` or the bmm/softmax pair, attention is your cost and this post's tricks apply; if it is the linear/conv layers, optimize those instead. **Two:** the gap between CPU time and CUDA time — a large CPU total with small CUDA total means you are *launch-bound* or *op-fallback-bound* (the device is idle waiting for the host), which on an edge accelerator usually means an unsupported op is running on the CPU and forcing round trips. **Three:** the shapes (`record_shapes=True`) — a kernel that is slow only at one shape often points at a layout the runtime did not fuse.

On a phone or a Jetson the equivalent tools are the vendor profilers: NVIDIA's Nsight Systems for Jetson (shows CPU/GPU/DLA placement and the dreaded fallback round trips), Android's `simpleperf` and the NNAPI/LiteRT delegate logs (which print exactly which ops the NPU accepted and which fell back to CPU), and Core ML's Instruments template (shows ANE vs GPU vs CPU per layer). The single most valuable thing these tools tell you is **op placement**: a transformer block that the NPU partially rejects will ping-pong tensors between NPU and CPU, and those transfers — not the math — become the latency. I have seen a MobileViT run 4× slower than its FLOPs predicted purely because the unfold/fold reshape ops were not supported on the NPU and every block round-tripped to the CPU. The fix was not a faster kernel; it was rewriting the reshape so the delegate would accept the whole block. You cannot find that without an op-placement trace.

#### Worked example: a profile that redirected a week of work

A concrete one from a real engagement. A team had a hybrid ViT detector missing its 30 ms p99 budget on a Jetson Orin Nano, running at ~52 ms. The instinct was "attention is quadratic, swap in linear attention." The roofline placement said otherwise: arithmetic intensity was *above* the ridge for most of the model, so it was compute-bound, not bandwidth-bound — linear attention (a bandwidth fix) would barely move it. The op trace then showed 40% of the time in a single transposed-conv upsampling head running on the CPU because the NPU delegate did not support that op variant. Replacing the transposed conv with a supported resize-plus-conv moved it onto the NPU and cut p99 to ~26 ms — *under budget, with zero change to attention.* The lesson is the section's whole point: the bottleneck is an empirical fact, not a guess from the architecture diagram, and the cheapest win is usually wherever the profile says the device is actually stalling.

## 10. Putting it together: an edge-attention checklist

The flow I follow when a transformer needs to fit on a device, in order:

1. **Profile first.** Where does the time and the memory actually go — attention or feed-forward, compute-bound or memory-bound? The [roofline](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) tells you whether to attack FLOPs or bytes.
2. **Switch to a fused exact kernel** (`F.scaled_dot_product_attention` with the flash backend, or your runtime's equivalent). Free, exact, often the single biggest win on long sequences. Confirm the backend actually fired.
3. **Reduce tokens before you reduce precision.** Token merging (training-free), pooling/downsampling, or a hybrid architecture that keeps token counts low. The cheapest token is the one you never compute.
4. **If still over budget, approximate attention** — windowed for vision, linear for diffuse long-context — knowing exactly what you traded.
5. **Prefer a hybrid (MobileViT/EfficientViT) over a pure ViT** for vision, and honestly ask whether a CNN already wins.
6. **Then quantize**, watching for activation outliers; keep LayerNorm/softmax in higher precision, calibrate on real data, escalate to smoothing/mixed-precision/QAT only as needed.
7. **Measure on the named target,** batch 1, warmed, with op-placement profiling — not FLOPs, not the workstation.

Each step composes with the next, which is the whole philosophy of the series: efficient architecture, then the other three levers, all validated by profiling and read off the accuracy–efficiency Pareto frontier. The capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) walks an end-to-end model through every lever; this post is the attention-and-ViT chapter of it.

## Key takeaways

- **Vanilla self-attention is $O(n^2 d)$ in compute and $O(n^2)$ in memory.** The $n \times n$ score matrix — per head, per layer — is what OOMs high-resolution and long-context models on the edge. Doubling tokens quadruples the cost.
- **Linear attention is the associativity reorder made legal by a kernel.** Choose $\text{sim}(q,k) = \phi(q)^\top\phi(k)$, build a small $d \times d$ state once, and the cost drops to $O(n d^2)$. It gives up *sharp, peaky* attention; great for diffuse global mixing, risky for hard retrieval.
- **FlashAttention is exact — same FLOPs, same output — and faster only because it moves less memory.** Tiling plus the online-softmax recurrence keeps the big matrix in SRAM, cutting HBM traffic from $O(n^2)$ to $O(n)$. Make it your first move; never approximate before you have removed memory-traffic waste.
- **It is a roofline story.** Attention is usually memory-bound; FLOP-reducing tricks do not help a memory-bound kernel. Profile to know which wall you are hitting.
- **Hybrids beat pure ViTs on the edge** because convolutions give locality and translation equivariance for free, cheaply, on the high-resolution early stages. MobileViT fuses local conv features with global transformer features; EfficientViT makes the attention itself linear.
- **The cheapest token is the one you never compute.** Token merging (training-free, ~2× throughput for a fraction of a point), pooling, and downsampling cut the sequence and so cut the quadratic.
- **A CNN often wins, and that is fine.** Reach for a transformer only when the task needs global context, the hybrid's *measured* latency fits, and the accuracy gain clears the bar.
- **Quantizing a ViT is not quantizing a CNN.** Activation outliers wreck naive int8; keep LayerNorm/softmax in higher precision, use per-channel/per-token (or smoothed) quantization, calibrate on real data.

## Further reading

- Vaswani et al. (2017), *Attention Is All You Need* — the original transformer and the $\text{softmax}(QK^\top/\sqrt{d})V$ that everything here optimizes.
- Katharopoulos et al. (2020), *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* — the kernel-feature linear attention derived in Section 2.
- Choromanski et al. (2021), *Rethinking Attention with Performers* — random-feature kernels that provably approximate softmax.
- Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (and FlashAttention-2, 2023) — the tiling + online-softmax IO-aware kernel.
- Liu et al. (2021), *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* — windowed attention for high-resolution vision.
- Mehta & Rastegari (2022), *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer* — the conv-transformer hybrid block.
- Cai et al. (2023), *EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention*; Bolya et al. (2023), *Token Merging: Your ViT But Faster*.
- PyTorch docs for `torch.nn.functional.scaled_dot_product_attention` and `torch.nn.attention.sdpa_kernel`; the `timm` model zoo for pretrained MobileViT/EfficientViT.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), [LLM quantization: activations, SmoothQuant, KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache), [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family), [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design), and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
