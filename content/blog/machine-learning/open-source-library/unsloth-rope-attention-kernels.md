---
title: "RoPE and Attention Kernels: Unsloth's Fused Rotary Embeddings"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "How Unsloth fuses rotary position embeddings into one in-place Triton kernel, runs the backward pass by negating the sine, and feeds a correctly laid-out context to Flash Attention instead of rewriting attention itself."
tags: ["unsloth", "rope", "rotary-embeddings", "attention", "flash-attention", "triton", "gqa", "llm-training", "kernel-fusion"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 28
---

Rotary position embeddings are the most-executed piece of arithmetic in a transformer that nobody talks about. Every attention layer rotates Q and K before the score matrix is computed. Every forward pass touches every layer. Every training step runs a forward and a backward. Multiply it out: a 32-layer model, 8192 tokens, batch of 4, and you are applying RoPE to roughly a million query-and-key vectors per step, then doing it all again on the way back. If RoPE is implemented as a string of PyTorch elementwise ops — and in the reference Hugging Face path, it is — then each of those ops allocates a tensor, reads it from HBM, writes it back, and gets thrown away. That is a lot of memory traffic for what is, mathematically, two multiplies and a sign flip.

This is the kind of hot, repetitive, allocation-heavy operation that fusion was invented for, and Unsloth fuses it about as tightly as it can be fused. The RoPE kernel rotates Q and K in place, processes heads in groups so the cosine/sine tables are loaded once and reused, and — the part that makes engineers smile when they first see it — runs the *exact same kernel* backward by negating the sine. A rotation matrix is orthogonal, so its transpose is just the negative-angle rotation. The gradient of "rotate by theta" is "rotate by minus theta," which is one line of code, not a second kernel.

![RoPE rotates each (q0, q1) pair by position angle m theta](/imgs/blogs/unsloth-rope-attention-kernels-1.webp)

The diagram above is the mental model: RoPE does not *add* anything to the query vector to mark its position. It *rotates* the vector. Position `m` becomes an angle `m * theta`, and each adjacent pair of channels `(q0, q1)` spins by that angle in its own little 2D plane. Because rotations preserve inner products and compose by adding angles, the dot product between a query at position `m` and a key at position `n` ends up depending only on the relative offset `m - n` — which is exactly the property you want a positional encoding to have. The whole mechanism is a rotation, and that single fact is what lets Unsloth's backward pass be so cheap.

This post is part of the [Inside Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) series. If you want the higher-level picture of how all of Unsloth's kernels add up to its speed and memory wins, read the [speedup anatomy](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) post first; for the general technique of melting several PyTorch ops into one Triton launch, see [Triton kernel fusion](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion). Here we go deep on one kernel — `_rope_embedding` — and on the honest story of what Unsloth does and does not do to attention.

## 1. RoPE is everywhere in the hot loop

Let me put numbers on "everywhere." Consider a fairly ordinary fine-tuning setup: Llama-3-8B has 32 transformer layers, 32 query heads and 8 key/value heads (it is a grouped-query model), and a head dimension of 128. Suppose you train at a sequence length of 4096 with a micro-batch of 2.

RoPE is applied to Q and to K in every layer. The Q tensor for one layer is `(batch, seq, n_heads, head_dim) = (2, 4096, 32, 128)`, which is about 33.5 million elements. The K tensor, with 8 heads, is about 8.4 million. So one layer rotates roughly 42 million elements on the forward pass. Across 32 layers that is about 1.3 *billion* element-rotations per forward pass, and the backward pass rotates the gradients of the same tensors, so double it. Per training step you are doing on the order of 2.7 billion rotations, and a real run is tens of thousands of steps.

Now ask what a "rotation" costs if you write it the naive way. The textbook formula — the one we will derive in the next section — is `RoPE(Q) = Q * cos + rotate_half(Q) * sin`. Read literally as PyTorch, that is: build `rotate_half(Q)` (a `cat` of two slices, which allocates a new tensor the size of Q), multiply `Q` by `cos` (allocate), multiply `rotate_half(Q)` by `sin` (allocate), add the two (allocate). Four-ish full-size intermediate tensors, each of which is written to HBM and read back, every layer, every step, for both Q and K.

The arithmetic intensity here is terrible. Each element gets touched by a couple of floating-point multiplies and then shipped across the memory bus several times. On modern GPUs the bottleneck for this kind of work is almost never the FLOPs; it is the HBM bandwidth and the kernel-launch overhead of all those separate elementwise ops. This is the textbook signature of an operation that wants to be fused: low compute per byte, high call frequency, lots of throwaway intermediates. Unsloth's answer is to do the entire formula in one Triton kernel that loads each value once, rotates it in registers, and stores it once — back over the input.

### Why this is worth a dedicated kernel

You might reasonably ask whether `torch.compile` would not just handle this. In principle a good fusing compiler can collapse the elementwise chain. In practice Unsloth deliberately keeps RoPE (and RMSNorm) *out* of `torch.compile` — the kernels are decorated with `@torch.compiler.disable` / `@torch._disable_dynamo` — because the hand-written Triton version is both faster and predictable, and because it has a hand-derived backward that the compiler would otherwise try to reconstruct through autograd. When an operation runs a billion times per step, you do not want to leave its performance to a heuristic. You write the kernel, you write its gradient, and you stop thinking about it.

## 2. RoPE in 90 seconds

Let me build the formula from the geometry, because once you see it as a rotation everything else falls out.

Take a query vector and split its `head_dim` channels into adjacent pairs: `(q0, q1), (q2, q3), ...`. Treat each pair as a point in a 2D plane. RoPE assigns each pair a frequency `theta_i` and, for a token at position `m`, rotates that pair by the angle `m * theta_i`. A 2D rotation by angle `phi` is the matrix

$$
R(\phi) = \begin{bmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{bmatrix}
$$

so for one pair the rotated values are

$$
q_0' = q_0 \cos\phi - q_1 \sin\phi, \qquad q_1' = q_1 \cos\phi + q_0 \sin\phi
$$

where `phi = m * theta_i`. Different pairs get different frequencies — the standard choice is `theta_i = base^(-2i/head_dim)` with `base = 10000` — so low-index pairs rotate slowly (capturing coarse, long-range position) and high-index pairs rotate fast (fine, local position). That spread of frequencies is what lets the model resolve position at many scales.

The reason this is a *good* positional encoding is the relative-position property. Because rotations compose additively, `R(m * theta) ^T R(n * theta) = R((n - m) * theta)`. When you take the attention score `q_m · k_n` between a rotated query at position `m` and a rotated key at position `n`, the absolute angles cancel and only the *difference* `n - m` survives. The model never sees an absolute position baked into the values; it sees relative offsets emerge naturally from the dot product. That is why RoPE extrapolates and interpolates to longer contexts more gracefully than a learned absolute embedding, and it is the foundation for the context-extension tricks covered in [long-context training](/blog/machine-learning/open-source-library/unsloth-long-context-training).

### The implementation identity

Doing a per-pair 2×2 matrix multiply is annoying to vectorize. The trick the whole ecosystem uses is to rewrite it as elementwise operations on the *whole* vector. Define `rotate_half(Q)` as: take the second half of the channels, negate it, and move it to the front, while the first half moves to the back. Concretely, for `head_dim = 4`, if `Q = [q0, q1, q2, q3]` then `rotate_half(Q) = [-q2, -q3, q0, q1]`. With cos and sin laid out so that each value lines up with its partner, the entire rotation collapses to

$$
\text{RoPE}(Q) = Q \odot \cos + \text{rotate\_half}(Q) \odot \sin
$$

where `⊙` is elementwise multiply.

![The RoPE identity decomposed into elementwise ops](/imgs/blogs/unsloth-rope-attention-kernels-2.webp)

The figure traces every operation. The only step that is not pointwise is `rotate_half` — the negate-and-swap — and it is exactly that step that the naive PyTorch implementation pays for with an allocation. Note also that `cos` and `sin` depend only on the position and the frequency, not on the head: every head at a given position rotates by the same angles, so the tables can be computed once and shared across all heads. Hold onto that fact; it is half of why the fused kernel is fast.

> A subtle layout point: this "split into two contiguous halves" convention (the GPT-NeoX / Llama style) is not the only one. The original RoPE paper interleaves pairs as `(q0, q1), (q2, q3)`. The two conventions are a permutation apart and are *not* interchangeable — a checkpoint trained with one will produce garbage if rotated with the other. Unsloth follows the half-split convention that Hugging Face Llama uses, which is why the kernel pairs index `i` with index `i + half_head_dim`.

## 3. The naive implementation

Here is the reference path, written the way you would find it in a stock Hugging Face attention module. This is *not* Unsloth code — it is the baseline Unsloth replaces, shown so the contrast is concrete.

```python
import torch

def rotate_half(x):
    # split the last dim in half, negate the second half, swap to the front
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)        # <-- allocates a full-size tensor

def apply_rope_naive(q, k, cos, sin):
    # cos, sin: (seq, head_dim), broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)        # (1, 1, seq, head_dim) — broadcast view
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)   # 3 elementwise ops + 1 cat, all new tensors
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

Count the allocations for `q` alone: `rotate_half(q)` allocates the `cat` result; `q * cos` allocates; `rotate_half(q) * sin` allocates; the final `+` allocates `q_embed`. That is four full-size tensors the size of Q, plus the same for K. Each is materialized in HBM, read back for the next op, and (for the intermediates) discarded. The broadcast of `cos`/`sin` is cheap as a *view*, but the multiplies still stream the full Q through memory.

None of these ops is expensive in isolation. The problem is that there are many of them, they all hit global memory, and they run a billion times. The fused kernel's entire job is to make the data cross the memory bus *once*.

![Naive HF RoPE vs Unsloth's fused kernel](/imgs/blogs/unsloth-rope-attention-kernels-3.webp)

The comparison above is the thesis of the whole post in one image. On the left, the naive path: a `cat` that allocates `rotate_half`, broadcast cos/sin, four or more elementwise ops each round-tripping HBM, a freshly allocated output tensor. On the right, the fused kernel: load the cos/sin slice once per row, rotate `q0` and `q1` in registers, two stores that write the result back over the input. Zero extra tensors. The numbers Unsloth quotes for its end-to-end training speedup are the *sum* of many fusions like this one; RoPE is one of the cleanest examples.

## 4. Unsloth's fused RoPE kernel

Now the real thing. Here is the Triton kernel from `unsloth/kernels/rope_embedding.py`, lightly trimmed (the full version carries a couple of extra dtype guards):

```python
import triton
import triton.language as tl

ROPE_GROUP_SIZE: int = 4

@triton.jit
def _rope_embedding(
    Q,     Q_row_stride,
    cos,   cos_row_stride,
    sin,   sin_row_stride,
    seqlen,
    head_dim:      tl.constexpr,
    n_heads:       tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE:    tl.constexpr,
):
    """RoPE is Q*cos + rotate_half(Q)*sin."""
    ROPE_GROUP_SIZE = 4
    row_position        = tl.program_id(0)        # which token / row
    group_head_position = tl.program_id(1)        # which group of heads
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # load the cos/sin slice for THIS row once; reused for every head in the group
    sin1 = tl.load(sin + (row_position % seqlen) * sin_row_stride + col_offsets,
                   mask=mask, other=0)
    cos1 = tl.load(cos + (row_position % seqlen) * cos_row_stride + col_offsets,
                   mask=mask, other=0)
    if BACKWARD_PASS:
        sin1 = -sin1                              # the transpose of the rotation = negate the sine

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end   = min((head_start + ROPE_GROUP_SIZE), n_heads)
    # one program rotates a GROUP of (up to) 4 heads — ~10% faster than 1 head/program (PR #238)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
```

Read it slowly, because almost every line is a deliberate optimization.

**The launch grid is 2D.** `tl.program_id(0)` is the row — one token's worth of one tensor, flattened so that `(batch, seq)` becomes a single row index. `tl.program_id(1)` is the head-group. The kernel is launched over `(n_rows, n_groups)`, so each program is responsible for one token and a contiguous block of four heads.

![The 2D launch grid: one program per (row, head-group)](/imgs/blogs/unsloth-rope-attention-kernels-4.webp)

**`half_head_dim` and the column mask.** The rotation pairs channel `i` with channel `i + half_head_dim`, so the kernel only ever indexes the *first* half with `col_offsets` and reaches the second half by adding `half_head_dim`. `BLOCK_SIZE` is chosen by `calculate_settings(head_dim // 2)`, so a single Triton block covers exactly half a head — the `mask = col_offsets < half_head_dim` discards any lanes past the real data. This is why a 128-dim head uses a 64-wide block: you process the two halves together, not the whole 128 in one sweep.

**Load cos/sin once, reuse for the whole group.** The `sin1`/`cos1` loads happen *before* the head loop. They are read once for the row and then reused for all four heads in the group. This is the payoff of the "cos/sin are shared across heads" observation from section 2 — and it is the single biggest reason head-grouping helps.

<figure class="blog-anim">
<svg viewBox="0 0 660 300" role="img" aria-label="One cos/sin table is loaded once per row and reused as a highlight sweeps across all four heads in the head-group" style="width:100%;height:auto;max-width:780px">
<title>A single cos/sin load is reused across all heads in the group</title>
<style>
.c2-table{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.c2-head{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.c2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.c2-code{font:600 13px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.c2-hi{fill:var(--accent,#6366f1);opacity:.16}
.c2-feed{stroke:var(--accent,#6366f1);stroke-width:2;fill:none}
@keyframes c2-sweep{0%,20%{transform:translateX(0)}25%,45%{transform:translateX(150px)}50%,70%{transform:translateX(300px)}75%,95%{transform:translateX(450px)}100%{transform:translateX(0)}}
.c2-anim{animation:c2-sweep 10s steps(1,end) infinite}
@keyframes c2-pulse{0%,100%{opacity:.5}50%{opacity:1}}
.c2-pulse{animation:c2-pulse 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.c2-anim{animation:none}.c2-pulse{animation:none;opacity:1}}
</style>
<rect class="c2-table" x="220" y="30" width="220" height="60" rx="8"/>
<text class="c2-lbl" x="330" y="55">cos1 / sin1</text>
<text class="c2-code" x="330" y="76">loaded once per row</text>
<path class="c2-feed c2-pulse" d="M270 90 L110 170"/>
<path class="c2-feed c2-pulse" d="M310 90 L270 170"/>
<path class="c2-feed c2-pulse" d="M360 90 L430 170"/>
<path class="c2-feed c2-pulse" d="M400 90 L590 170"/>
<rect class="c2-head" x="40"  y="170" width="130" height="70" rx="8"/>
<rect class="c2-head" x="200" y="170" width="130" height="70" rx="8"/>
<rect class="c2-head" x="360" y="170" width="130" height="70" rx="8"/>
<rect class="c2-head" x="520" y="170" width="130" height="70" rx="8"/>
<rect class="c2-hi c2-anim" x="40" y="170" width="130" height="70" rx="8"/>
<text class="c2-lbl" x="105" y="200">head 0</text>
<text class="c2-lbl" x="265" y="200">head 1</text>
<text class="c2-lbl" x="425" y="200">head 2</text>
<text class="c2-lbl" x="585" y="200">head 3</text>
<text class="c2-code" x="105" y="222">rotate</text>
<text class="c2-code" x="265" y="222">rotate</text>
<text class="c2-code" x="425" y="222">rotate</text>
<text class="c2-code" x="585" y="222">rotate</text>
<text class="c2-code" x="330" y="280">ROPE_GROUP_SIZE = 4 heads share one cos/sin load</text>
</svg>
<figcaption>The cos/sin tables are loaded once per row, then the program loops over all four heads in the group; the highlight marks the head being rotated while the shared tables stay resident.</figcaption>
</figure>

**The `row_position % seqlen` indexing.** The Q tensor is flattened so a row index runs over `batch * seq`, but the cos/sin tables are only `seqlen` long — one entry per position, not per (batch, position). The modulo maps a flattened row back to its position within the sequence so the right angle is fetched regardless of which batch element it belongs to. It is a small thing, but get it wrong and every batch element after the first rotates by the wrong angle.

**The rotation, in registers, written in place.** Inside the head loop, `Q1` and `Q2` are the two halves. The two stores write `Q1 * cos1 - Q2 * sin1` back to `offs_q1` and `Q2 * cos1 + Q1 * sin1` back to `offs_q2` — exactly the per-pair rotation from section 2, applied to the whole half-block at once. The values are loaded, rotated in registers, and stored back over their own addresses. No scratch tensor, no `rotate_half` allocation, no separate output buffer.

![In-place rotation across the two halves of head_dim](/imgs/blogs/unsloth-rope-attention-kernels-5.webp)

### Why ROPE_GROUP_SIZE = 4

The choice of four heads per program is empirical. Unsloth's commit history (the change is attributed to PR #238) reports it as roughly 10% faster than the original one-head-per-program version. The intuition: a single head's half-block is small — 64 elements for a 128-dim head — which under-uses the program's registers and amortizes the launch and the cos/sin load poorly. Grouping four heads lets one program reuse the same `cos1`/`sin1` across four rotations and keeps the GPU's pipelines fuller, without making the group so large that register pressure or tail effects (the `min(..., n_heads)` clamp handles a ragged last group) start to hurt. It is a classic occupancy-vs-reuse tuning knob, and four is where it landed for the head dimensions transformers actually use.

## 5. The backward trick

Here is the part worth the price of admission. The autograd wrapper:

```python
class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.reshape(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = Q.shape
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)
        with torch_gpu_device(Q.device):
            _rope_embedding[(n_rows, n_groups,)](
                Q, Q.stride(0), cos, cos.stride(0), sin, sin.stride(0),
                seq_len, head_dim, n_heads,
                BACKWARD_PASS=False, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.n_groups   = n_groups
        ctx.cos = cos                  # cos/sin cached on ctx — NOT recomputed in backward
        ctx.sin = sin
        return Q.reshape(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = dY.shape
        cos = ctx.cos
        sin = ctx.sin
        with torch_gpu_device(dY.device):
            _rope_embedding[(n_rows, ctx.n_groups,)](
                dY, dY.stride(0), cos, cos.stride(0), sin, sin.stride(0),
                seq_len, head_dim, n_heads,
                BACKWARD_PASS=True, BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
        dY = dY.reshape(batch, seq_len, n_heads, head_dim)
        return dY, None, None
```

Look at what the backward pass does. It calls *the same kernel* — `_rope_embedding` — with `BACKWARD_PASS=True`. That flag does exactly one thing inside the kernel: `sin1 = -sin1`. The forward applies `R(theta)`; the backward applies `R(-theta)`. There is no separate backward kernel, and the gradient `dY` is rotated in place exactly the way the input was.

### Why negating the sine is the gradient

This is not a coincidence or a clever approximation — it is the calculus, and it is exact. RoPE on one pair is a linear map: `q' = R(theta) q`, where `R(theta)` is the 2×2 rotation matrix. For a linear map `y = A x`, the backward pass (vector-Jacobian product) is `dx = A^T dy`. So the gradient flowing back through RoPE is `R(theta)^T` applied to the incoming gradient.

A rotation matrix is *orthogonal*: `R(theta)^T = R(theta)^{-1} = R(-theta)`. Negating the angle of a rotation gives you both its inverse and its transpose at once. So the backward map is `R(-theta)`, and the only difference between `R(theta)` and `R(-theta)` is the sign of the off-diagonal `sin` terms. Flip the sign of `sin` and you have turned the forward rotation kernel into its own gradient.

<figure class="blog-anim">
<svg viewBox="0 0 640 320" role="img" aria-label="A query vector rotates forward by plus theta in the forward pass, then unrotates by minus theta in the backward pass, returning to its original direction" style="width:100%;height:auto;max-width:760px">
<title>Forward rotation by +theta, backward rotation by -theta, returns to start</title>
<style>
.r1-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.r1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.r1-sub{font:600 13px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.r1-ghost{stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:5 5}
.r1-vec{stroke:var(--accent,#6366f1);stroke-width:4}
.r1-grp{transform-box:fill-box;transform-origin:center}
@keyframes r1-spin{0%,8%{transform:rotate(0deg)}42%,58%{transform:rotate(-55deg)}92%,100%{transform:rotate(0deg)}}
.r1-anim{animation:r1-spin 9s ease-in-out infinite}
@keyframes r1-fwd{0%,8%{opacity:.25}30%,58%{opacity:1}80%,100%{opacity:.25}}
@keyframes r1-bwd{0%,42%{opacity:.25}64%,86%{opacity:1}100%{opacity:.25}}
.r1-fwd{animation:r1-fwd 9s ease-in-out infinite}
.r1-bwd{animation:r1-bwd 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.r1-anim{animation:none}.r1-fwd,.r1-bwd{animation:none;opacity:1}}
</style>
<line class="r1-axis" x1="60" y1="250" x2="580" y2="250"/>
<line class="r1-axis" x1="320" y1="40" x2="320" y2="270"/>
<text class="r1-sub" x="560" y="275">q0</text>
<text class="r1-sub" x="300" y="55">q1</text>
<line class="r1-ghost" x1="320" y1="250" x2="500" y2="250"/>
<text class="r1-lbl" x="500" y="240">original q</text>
<g class="r1-grp r1-anim">
<line class="r1-vec" x1="320" y1="250" x2="500" y2="250"/>
<polygon points="500,250 484,242 484,258" fill="var(--accent,#6366f1)"/>
</g>
<text class="r1-lbl r1-fwd" x="160" y="300">forward: R(+theta) = q*cos + rotate_half(q)*sin</text>
<text class="r1-lbl r1-bwd" x="480" y="300">backward: R(-theta) negates sin, same kernel</text>
</svg>
<figcaption>The forward pass rotates the vector by +theta; the backward pass runs the identical kernel with sin negated, applying R(-theta) and landing exactly back on the original direction.</figcaption>
</figure>

There is a second, quieter win in that autograd wrapper: `ctx.cos = cos; ctx.sin = sin`. The cosine and sine tables are *cached on the context object*, not recomputed in the backward pass and not even saved through the usual `save_for_backward` machinery that would tie them into the autograd graph. RoPE's cos/sin depend only on the positions and frequencies, which are fixed for the whole batch, so there is no reason to regenerate them. Many naive implementations recompute `cos`/`sin` (or re-fetch a large precomputed buffer) on the backward pass; Unsloth just hands the same two tensors back to the same kernel. Combined with the in-place rotation, the backward pass for RoPE costs almost exactly what the forward pass costs: two loads, two multiplies per element, two stores, and not one byte of extra allocation.

This pattern — a hand-derived backward that reuses the forward kernel and saves only what is cheap to save — is the same philosophy you see across Unsloth's kernels (the RMSNorm backward reuses the saved `1/rms`; the cross-entropy backward recomputes softmax from a per-row logsumexp). It is the subject of the [manual backprop](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion) work more broadly. RoPE is the most elegant instance because the math hands you the gradient for free.

## 6. GQA and grouped heads

A modern detail the kernel handles without fuss: Q and K usually have a *different* number of heads. Grouped-query attention (GQA) — used by Llama-3, Mistral, Qwen, and most recent models — keeps 32 query heads but only 8 key/value heads, so several query heads share one KV head. That cuts the KV cache by 4× at inference and the K/V projection cost at training, with little quality loss.

For RoPE this is a non-issue *because the kernel is parameterized by `n_heads`*. The forward computes `n_groups = ceil(n_heads / ROPE_GROUP_SIZE)` from whatever head count it is given. Rotate Q and you pass `n_heads = 32`; rotate K and you pass `n_heads = 8`. Same kernel, different grid. The wrapper that ties it together rotates both:

```python
def fast_rope_embedding(Q, K, cos, sin):
    # Q: (batch, seq, n_q_heads, head_dim),  K: (batch, seq, n_kv_heads, head_dim)
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K
```

Two independent calls, one for the query stream and one for the key stream, each launching `_rope_embedding` over its own `(n_rows, n_groups)` grid. The cos/sin tables are *identical* for both — position `m` rotates the query at `m` and the key at `m` by the same angles — so the same two tensors are passed to both calls. There is nothing GQA-specific in the rotation math; the head count just changes how many programs the grid spawns.

> **Multi-GPU note.** When Unsloth runs across multiple GPUs, kernel launches are wrapped in `torch_gpu_device(Q.device)` so each program is enqueued on the stream of the device that owns the tensor, and stream synchronization is handled around the launch. Earlier write-ups (including the older [overview post](/blog/machine-learning/open-source-library/unsloth-lib)) said Unsloth did not support multi-GPU; that is no longer true — multi-GPU is available now, with a larger upgrade in progress. The RoPE kernel itself is device-local and needs no change to run on each GPU's shard.

## 7. Attention itself: Unsloth does not write the attention kernel

This is the section where most explanations of Unsloth quietly mislead, so let me be blunt. Unsloth does **not** ship a custom attention kernel. It does not reimplement `softmax(QK^T / sqrt(d)) V`. The actual attention computation — the part with the quadratic score matrix and the online softmax — is delegated to **Flash Attention 2** (or xformers' memory-efficient attention, depending on what is installed and what the hardware supports).

![Where fused RoPE sits in the attention path](/imgs/blogs/unsloth-rope-attention-kernels-6.webp)

What Unsloth owns is everything *around* attention, as the figure lays out:

- The **QKV projection**, fused with LoRA and 4-bit dequantization (covered in the manual-backprop LoRA work).
- The **fused RoPE** we just dissected.
- The **tensor layout** — keeping Q, K, V contiguous in the `(batch, n_heads, seq, head_dim)` shape that Flash Attention wants, so there is no redundant transpose or copy sitting between the projection and the attention call.

Then it hands those tensors to Flash Attention and lets it do the math it is already the best in the world at. Flash Attention still does the `O(seq^2)` score computation; it still uses the tiled, online-softmax trick that never materializes the full `seq × seq` score matrix in HBM. Unsloth does not, and could not easily, beat that — Flash Attention is a deeply tuned, hardware-specific kernel maintained by people who do nothing else.

Why does this matter for how you read Unsloth's benchmarks? Because if you believed Unsloth had a magic attention kernel, you would expect its win to come from the attention itself, and you would be surprised when profiling shows attention taking the same time as a plain Flash Attention 2 baseline. It does take the same time — *that is the point*. The win is upstream and downstream: the projection is fused, the RoPE is fused, the layout is correct so no copy is inserted, the gradient checkpointing offloads activations, the cross-entropy doesn't blow up memory. Attention is a fixed cost that Unsloth pays the same as everyone else and then surrounds with much cheaper everything-else.

## 8. Reading the "10–30x" benchmark honestly

Unsloth's marketing and the broader conversation around it have, at various times, thrown around numbers like "10x faster on a single GPU" and "30x faster on multi-GPU." You should understand exactly what those compare before you quote them, because the honest version is still impressive and the dishonest version will get you in trouble in a design review.

First, the current, conservative headline numbers from Unsloth's own README (2026) are more measured: "up to 2x faster with up to 70% less VRAM" for typical fine-tuning, "80% less VRAM" for GRPO, "train MoE LLMs 12x faster with 35% less VRAM," and "3x faster training and 30% less VRAM" from the newer Triton kernels and padding-free packing. Those are the numbers to anchor on.

The larger "10–30x" figures, where they appear, are **end-to-end fine-tuning throughput** measured against a *naive* Hugging Face baseline — and crucially, often a baseline that was *not* itself using Flash Attention, or was using it with suboptimal settings, plus full-precision optimizer states, no gradient checkpointing tuning, and unfused everything. When the comparison is "Unsloth's whole stack" versus "stock `transformers` with defaults," the multiplier is large because the baseline is leaving an enormous amount on the table across many axes at once.

The honest framing is this: **Unsloth's speedup is the product of many fusions, not a single magic kernel — and certainly not a faster attention.** Stack them up and they compound:

| Source of the win | What it replaces | Roughly where it shows up |
| --- | --- | --- |
| Fused RoPE (this post) | 4+ elementwise ops + `cat`, per Q/K, per layer | memory bandwidth, kernel launches |
| Fused RMSNorm | square/mean/rsqrt/scale as separate ops | memory bandwidth |
| Manual-backprop LoRA + 4-bit | autograd graph + fp16 base weights resident | VRAM, matmul count |
| Fused cross-entropy | full `(batch·seq, vocab)` softmax tensor | VRAM (huge for big vocabs) |
| Offloaded gradient checkpointing | activations resident across full depth | VRAM |
| Correct contiguous layout | redundant transpose/copy before attention | bandwidth, latency |
| Flash Attention 2 (delegated) | — (same as a good baseline) | unchanged |

Notice the last row. Attention is *unchanged*. If your mental model of "why Unsloth is fast" had attention in it, replace that slot with "everything around attention, fused." When you benchmark Unsloth against a *properly configured* baseline — one that already uses Flash Attention 2, bf16, 4-bit QLoRA, and gradient checkpointing — the gap narrows to the "up to 2x, up to 70% less VRAM" range the README now quotes, because at that point you are measuring the value of the fusions alone, not the value of fixing a badly configured baseline. Both numbers are real; they answer different questions. Quote the 2x against a strong baseline and the larger numbers only with the "versus naive HF defaults" caveat attached.

And remember the other half of Unsloth's pitch, which the fusions make possible without compromise: **zero accuracy loss**. The kernels are exact rewrites, not approximations. The fused RoPE computes the same `Q*cos + rotate_half(Q)*sin` to the same precision as the naive path; the backward is the exact transpose, not a finite-difference estimate. The speed comes from doing the same arithmetic with less memory traffic, not from doing less arithmetic.

## 9. Case studies: where this kernel earns its keep

Abstract performance arguments are easy to wave at. Here are concrete situations — drawn from the kinds of issues that actually come up when people fine-tune — where the design decisions in `_rope_embedding` change the outcome.

### 9.1 The long-context run that OOMs on the naive path

A team fine-tunes Llama-3-8B at 8192 tokens to teach it to summarize long documents. On the stock path, every layer's RoPE allocates `rotate_half(Q)`, `Q*cos`, `rotate_half(Q)*sin`, and the sum — four tensors of shape `(batch, 8192, 32, 128)` for Q, plus the K versions, materialized and freed every layer. The transient allocations spike the memory high-water mark right when activations are also at their peak, and the run OOMs at batch 2. Switching to Unsloth's in-place RoPE removes those transients entirely: the rotation happens in registers and overwrites the input, so the RoPE step contributes *zero* to the activation high-water mark. The same hardware now fits batch 2 comfortably. This is the unglamorous reason fused kernels matter — not throughput, but headroom.

### 9.2 The grouped-query model where the head counts didn't line up

An engineer ports a custom attention module to a GQA model and writes their own RoPE that hardcodes `n_heads` from the query tensor, then applies the same call to K. K has a quarter as many heads, so the code either indexes out of bounds or silently rotates the wrong slices. The bug is subtle because the loss still goes down — just slower and to a worse minimum, because the keys are being positionally encoded incorrectly. Unsloth's kernel sidesteps the whole class of bug by taking `n_heads` as a `constexpr` argument and computing the grid from it per call, so `fast_rope_embedding(Q, K, cos, sin)` rotates the 32-head Q and the 8-head K each with the right grid. The lesson generalizes: parameterize by the head count, never assume Q and K match.

### 9.3 The `torch.compile` regression nobody could explain

A team enables `torch.compile` on their training loop expecting a speedup and instead gets a slowdown and occasional numerical drift in the gradients. After a long bisection they find the compiler had folded the RoPE elementwise chain into a fused region but reconstructed the backward through autograd in a way that recomputed cos/sin and changed the reduction order. Unsloth avoids this entirely by decorating its RoPE and RMSNorm kernels with `@torch.compiler.disable` / `@torch._disable_dynamo` — these ops are explicitly fenced off from the compiler because the hand-written kernel with its hand-derived backward is both faster and bit-stable. The takeaway: a hand-tuned kernel with an exact manual gradient is sometimes better left *out* of the compiler's reach, and Unsloth makes that call deliberately.

### 9.4 The backward pass that was twice as expensive as it needed to be

A from-scratch RoPE implementation defines a `torch.autograd.Function` whose backward recomputes `cos` and `sin` from the position ids and frequency base, then builds a *second* set of intermediate tensors to apply the inverse rotation. Profiling shows the RoPE backward taking nearly twice the time of the forward. The fix is the one Unsloth ships: cache `cos`/`sin` on `ctx` so they are never recomputed, and reuse the forward kernel with `sin -> -sin` so the inverse rotation needs no new code and no new buffers. The backward collapses to the cost of the forward. When you see a backward pass costing noticeably more than its forward for a *linear, orthogonal* op, suspect a recompute or an allocation you don't need.

### 9.5 The fine-tune that broke because of the wrong RoPE convention

An engineer adapts a model checkpoint by swapping in an attention implementation that uses the *interleaved* RoPE convention (`(q0,q1),(q2,q3)` pairs) when the checkpoint was trained with the *half-split* convention (first half paired with second half). Numerically the rotation is "applied," so nothing crashes, but the model's learned positional structure is permuted into nonsense and quality collapses. Unsloth follows the half-split convention that matches Hugging Face Llama checkpoints, pairing index `i` with `i + half_head_dim` exactly as the upstream model expects. The broader point: RoPE is not one operation but a *family* differing by channel layout, and the kernel must match the convention the weights were trained under, byte for byte.

### 9.6 The multi-GPU launch that raced on the wrong stream

When Unsloth gained multi-GPU support, an early failure mode was kernels enqueued on the default CUDA stream while the tensors lived on a non-default device, producing intermittent wrong results that only appeared under load. The fix visible in the current code is the `torch_gpu_device(Q.device)` context manager wrapping every launch, which pins the kernel to the stream of the device that owns the data. For a device-local op like RoPE this is all that is needed — the rotation never crosses devices — but it has to be correct, because a positional encoding that is occasionally wrong is far worse than one that is consistently wrong: the model averages over the noise and learns nothing useful from those tokens.

## When to reach for this — and when not to

**Reach for Unsloth's fused RoPE when** you are fine-tuning a supported architecture (Llama, Mistral, Qwen, Gemma, and friends) and you want the memory headroom and throughput without touching the math. You get an exact, in-place rotation with a free backward and no accuracy cost. If you are training at long context or at the edge of your VRAM budget, the elimination of RoPE's transient allocations alone can be the difference between fitting and OOMing.

**Reach for it as a pattern to copy when** you are writing your own kernels for any linear, structure-preserving operation. The two ideas generalize far beyond RoPE: (1) if your forward is a linear map `A`, your backward is `A^T` — and if `A` is orthogonal, `A^T` is just `A` with a sign or index flip, so reuse the kernel; (2) anything that depends only on fixed inputs (position, frequency) should be computed once and cached on `ctx`, never recomputed in backward.

**Do not reach for it when** you need a RoPE *variant* the kernel doesn't implement — exotic frequency schedules, partial rotation over only some channels, or the interleaved convention for a checkpoint trained that way. The kernel is fast precisely because it assumes the half-split layout and the standard frequency formula; if your model needs something else, you either extend the kernel (carefully, keeping the backward exact) or fall back to a correct PyTorch implementation and eat the allocations. And do not reach for Unsloth expecting it to make *attention itself* faster — it won't, because it delegates attention to Flash Attention. What it makes faster is everything around attention, RoPE very much included.

The deeper lesson is the one the whole [Inside Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) series keeps returning to. There is rarely a single trick. There is a hot operation that runs a billion times, and someone who refused to accept that it had to allocate four tensors and launch four kernels to do two multiplies and a sign flip. RoPE is the cleanest example because the geometry is so kind — a rotation is orthogonal, so its gradient is the same rotation backward — but the discipline is the same one applied to RMSNorm, to cross-entropy, to LoRA, to checkpointing. Fuse the hot loop. Save only what's cheap. Let the specialists (Flash Attention) own what they own. The speedups compound.
