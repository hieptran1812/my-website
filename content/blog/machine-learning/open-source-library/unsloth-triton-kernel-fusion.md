---
title: "Fused Triton Kernels: How Unsloth Rewrites RMSNorm and SwiGLU"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "Most fine-tuning ops are memory-bandwidth-bound, not compute-bound. This is how Unsloth fuses each normalization and activation into a single Triton kernel that keeps data on-chip, cutting both launch overhead and HBM traffic with zero change to the math."
tags: ["unsloth", "triton", "gpu-kernels", "kernel-fusion", "rmsnorm", "swiglu", "memory-bandwidth", "qlora", "fine-tuning", "cuda"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 28
---

Here is the fact that surprises most people the first time they profile a fine-tuning run: the GPU spends a startling fraction of its time doing almost no arithmetic. Multiply two big matrices and the tensor cores are saturated — that is compute-bound work, and it is the part everyone optimizes. But a transformer block is not all matmuls. Between the projections sit RMSNorm, RoPE, the SwiGLU activation, residual adds, dropout — a long tail of *elementwise* and *reduction* operations. Each one reads a big activation tensor out of high-bandwidth memory (HBM), does a trivial amount of math on it, and writes the result straight back. The arithmetic is free. The memory traffic is the entire cost.

[Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) is built almost entirely around this observation. Its headline numbers — "3x faster training and 30% less VRAM via new Triton kernels and padding-free packing," and broadly "up to 2x faster with up to 70% less VRAM" — do not come from a clever approximation or a lower-precision shortcut. The kernels are *exact* rewrites; QLoRA numerics still match bitsandbytes bit-for-bit. The speed comes from refusing to pay the memory tax. Where naive PyTorch launches half a dozen tiny kernels that each round-trip the activation through HBM, Unsloth fuses the whole layer into **one** Triton kernel that keeps every intermediate in registers and SRAM, and writes back to HBM exactly once.

![Naive layer vs fused kernel: a naive RMSNorm round-trips activations through HBM five times; the fused kernel reads once and writes once.](/imgs/blogs/unsloth-triton-kernel-fusion-1.webp)

The diagram above is the mental model for the whole post. On the left, the naive layer: a tall column of HBM, and an arrow leaving it for every elementwise step, because every step is its own kernel and every kernel reads its input from HBM and writes its output back. On the right, the fused kernel: one read, one write, and a green box in between where `square → mean → rsqrt → scale` all happen without a single intermediate ever touching HBM. Same math. Same output, to the last bit. A fraction of the memory traffic and a fraction of the launch overhead.

This post is a close reading of two of those fused kernels — RMSNorm and SwiGLU — straight from Unsloth's source. We will count the kernel launches and the bytes moved in the naive version, then walk the Triton rewrite line by line and show exactly where each intermediate lives. If you have read the [speedup anatomy](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) post, this is the zoomed-in view of one of its biggest levers; the [manual-backprop](/blog/machine-learning/open-source-library/unsloth-manual-backprop) post is the matching story for the backward pass.

## 1. Why fine-tuning is memory-bound, not compute-bound

Start with intuition before any code. Picture a short-order kitchen. The expensive, skilled work is cooking — that is your matmul, the thing that genuinely keeps the tensor cores (the chefs) busy. But between every cooking step, a runner carries the dish back to the pantry and fetches it again for the next station. If each station is a separate trip to the pantry, the chefs spend most of the night standing idle while the runner sprints back and forth. The kitchen is not *cooking-bound*; it is *runner-bound*. Fusing the stations — chop, sear, plate in one place without a pantry trip in between — does not make the cooking faster, but it eliminates the runner, and the runner was the bottleneck.

On a GPU, the "runner" is HBM bandwidth and the "pantry trips" are kernel launches that read and write global memory. The reason this matters so much for *fine-tuning* in particular is the operation mix. A LoRA or QLoRA fine-tune freezes the big weight matrices, so the expensive dense matmuls are either cheap (small batch) or quantized. What is left dominating the wall-clock is the elementwise and normalization tail, and that tail is pure memory traffic. Optimize the matmul all you want; if RMSNorm and SwiGLU are launching a dozen bandwidth-bound kernels per layer, that is where your time goes.

The quantitative way to say this is **arithmetic intensity**: the ratio of floating-point operations performed to bytes moved to and from HBM, in FLOPs per byte. Every GPU has a *ridge point* — the arithmetic intensity at which it transitions from memory-bound to compute-bound. For an H100 with roughly 3.3 TB/s of HBM bandwidth and on the order of 1,000 TFLOP/s of usable bf16 throughput, that ridge sits around 300 FLOPs per byte. Anything below that is starved for bytes; the math units idle waiting on memory.

Now look at where the operations in a transformer block fall:

| Operation | FLOPs per element | Bytes per element (bf16) | Arithmetic intensity | Bound by |
| --- | --- | --- | --- | --- |
| Dense matmul (large) | $O(\text{inner dim})$ | small (weights reused) | hundreds–thousands | compute |
| RMSNorm | ~5 | read 2 + write 2 = 4 | ~1 | memory |
| SwiGLU (SiLU × gate) | ~5 | read 4 + write 2 = 6 | <1 | memory |
| Residual add | 1 | read 4 + write 2 = 6 | ~0.2 | memory |
| RoPE | ~6 | read 2 + write 2 = 4 | ~1.5 | memory |

Every non-matmul row sits at an arithmetic intensity of roughly 1 — two to three orders of magnitude below the ridge. These ops *cannot* keep the math units busy no matter how fast the GPU computes, because they are limited entirely by how fast bytes arrive. The instant an op is memory-bound, the only knob that matters is **bytes moved**. Reduce the bytes and you reduce the time, linearly. That is the entire game, and fusion is how you reduce the bytes.

There is a second tax stacked on top of the bandwidth tax: **kernel launch latency**. Every CUDA kernel launch is a host-to-device round-trip — the CPU enqueues the launch, the driver and hardware schedule it, and there is a fixed overhead on the order of a few microseconds per launch before a single thread runs. For a big matmul that takes milliseconds, a few microseconds is rounding error. For a tiny elementwise kernel that *also* takes a few microseconds, the launch overhead can rival or exceed the actual work. A naive RMSNorm fires five of these in a row; multiply by the number of layers and the number of normalizations per layer and you are paying launch latency thousands of times per forward pass. Fusion collapses those launches too.

So the thesis, stated precisely: **fine-tuning's non-matmul ops are memory-bandwidth-bound and launch-overhead-bound; fusion attacks both at once by keeping intermediates on-chip and collapsing many launches into one.** The rest of this post is the mechanism.

## 2. What a GPU kernel actually costs

To see why fusion wins, you have to know where data lives and how fast each tier moves. The GPU memory hierarchy is a steep pyramid, and the whole point of a fused kernel is to climb it.

![The GPU memory hierarchy: registers and SRAM are roughly 100x faster than HBM, so keeping intermediates on-chip is the win.](/imgs/blogs/unsloth-triton-kernel-fusion-2.webp)

From the bottom up:

- **HBM / global memory** is the big off-chip DRAM — 80 GB on an H100, with ~3.3 TB/s of bandwidth and a latency of *hundreds* of nanoseconds for an uncached access. This is where your weights, activations, and every PyTorch tensor live. It is enormous and, relative to the compute units, agonizingly slow. This is the wall.
- **L2 cache** is chip-wide, a few tens of MB, several TB/s, tens of nanoseconds. It catches some reuse automatically but you do not program it directly.
- **SRAM / shared memory** is per-streaming-multiprocessor (per-SM), on the order of a hundred-odd KB per SM, with bandwidth in the high teens of TB/s and single-digit-nanosecond latency. You *do* control this: it is the scratchpad a kernel uses for block-level reductions and tile staging.
- **Registers** are per-thread, the fastest storage on the chip — tens of TB/s aggregate, roughly one-cycle access. A kernel's working set, if it fits, lives here.

The bandwidth gap between registers/SRAM and HBM is the crux: on-chip storage is roughly *two orders of magnitude* faster. An intermediate value that lives and dies in registers costs effectively nothing to produce and consume. The same value, if it has to be written to HBM and read back, costs a full HBM round-trip — and at arithmetic intensity ~1, that round-trip *is* the operation's runtime.

Here is the consequence that drives kernel design. When PyTorch executes `y = x.pow(2)` in eager mode, it launches a kernel that:

1. reads the entire tensor `x` from HBM into registers,
2. squares each element (one cheap FLOP),
3. writes the entire result back to HBM.

The squared tensor now sits in HBM. The next op, `.mean(-1)`, launches *another* kernel that reads that squared tensor back from HBM, reduces it, and writes the mean to HBM. And so on. Every `torch` operation is a fence: it must fully materialize its output in global memory before the next operation can begin, because the next operation is a separate kernel that knows nothing about registers from the previous one. The intermediate values bounce off HBM at every step.

A **fused** kernel breaks the fence. It does the square, the mean, the rsqrt, and the scale *within a single kernel invocation*, so the intermediate values never leave registers. The only HBM traffic is reading the original input once and writing the final output once. Everything in between happens at register speed. That is the difference between arithmetic intensity ~1 (read-compute-write per op) and arithmetic intensity ~5/4 for the whole fused layer — still memory-bound, but moving a third of the bytes and paying one launch instead of five.

[Triton](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive) is the tool Unsloth uses to write these fused kernels. It is a Python-embedded language that compiles to GPU code, where you express a kernel as a grid of *programs* — one program per independent unit of work — and within each program you load from HBM into on-chip tensors, compute, and store back. The programmer controls exactly what crosses the HBM boundary. That control is the whole reason Unsloth can hand-fuse a layer that PyTorch's eager mode necessarily splits apart.

## 3. RMSNorm the naive way: five launches, four wasted tensors

RMSNorm (root-mean-square normalization) is the normalization used in Llama, Mistral, Gemma, and most modern decoder LLMs. It is simpler than LayerNorm — no mean subtraction, no bias — but its naive implementation is a textbook memory-traffic disaster. Here is what a faithful eager PyTorch RMSNorm looks like:

```python
import torch
import torch.nn as nn

class NaiveRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # x: (batch, seq, hidden), e.g. (4, 2048, 4096)
        squared   = x.pow(2)                       # kernel 1: write sq[b, s, h]
        mean_sq   = squared.mean(-1, keepdim=True) # kernel 2: write ms[b, s, 1]
        inv_rms   = torch.rsqrt(mean_sq + self.variance_epsilon)  # kernel 3: write ir[b, s, 1]
        normed    = x * inv_rms                     # kernel 4: write nm[b, s, h]
        return normed * self.weight                 # kernel 5: write y[b, s, h]
```

The math is exactly right: $\text{RMSNorm}(x) = \dfrac{x}{\sqrt{\frac{1}{H}\sum_{i} x_i^2 + \epsilon}} \odot w$, where $H$ is the hidden dimension and $w$ is the learned per-channel scale. But count the kernels and the tensors.

![Naive RMSNorm: each PyTorch op is a separate kernel launch that reads and writes a full activation tensor to HBM; four temporaries are pure overhead.](/imgs/blogs/unsloth-triton-kernel-fusion-3.webp)

Five operations, five kernel launches. And every one of them reads its input from HBM and writes its output to HBM:

- **`x.pow(2)`** reads `x` (full `b·s·h` activation), writes `squared` (same size) to HBM.
- **`.mean(-1)`** reads `squared` back, writes `mean_sq` (small, `b·s·1`).
- **`rsqrt(... + eps)`** reads `mean_sq`, writes `inv_rms` (small).
- **`x * inv_rms`** reads `x` *again* and broadcasts `inv_rms`, writes `normed` (full size).
- **`normed * weight`** reads `normed`, writes `y` (full size).

The four intermediate tensors — `squared`, `mean_sq`, `inv_rms`, `normed` — are pure overhead. Nobody wants them. They exist only because eager mode has to materialize each step's output before the next step can start. For the full-sized intermediates (`squared` and `normed`), each is a full activation tensor allocated in HBM, written, and read back. On a `(4, 2048, 4096)` bf16 activation that is 64 MB *per intermediate tensor*, allocated and round-tripped, for a result we could have computed in one pass.

This is not a strawman. It is exactly what `nn.Module`-style RMSNorm does in eager mode, and it is why every serious training stack — Unsloth, the [TRL](/blog/machine-learning/open-source-library/trl-lib) fast paths, [torchtitan](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive) — replaces it with a fused kernel. PyTorch's own `torch.compile` will fuse this pattern too, and we will come back to *why Unsloth deliberately does not let `torch.compile` near these kernels* in section 7. For now, the takeaway is the count: **5 launches, ~6 HBM passes over the full activation, 4 unwanted tensors.**

## 4. RMSNorm the Unsloth way: one kernel, one row at a time

Here is Unsloth's forward kernel, verbatim from `unsloth/kernels/rms_layernorm.py`:

```python
import triton
import triton.language as tl
import torch
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride: tl.constexpr,
    X, X_row_stride: tl.constexpr,
    W, W_row_stride: tl.constexpr,
    r, r_row_stride: tl.constexpr,
    n_cols: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)                 # save 1/rms for the backward pass
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)
```

Read it slowly, because every line is load-bearing.

**One program per row.** The kernel is launched with a grid of `(n_rows,)` — one program instance for each row of the flattened `(n_rows, n_cols)` activation, where `n_cols` is the hidden dimension and `n_rows` is `batch × seq`. The very first line, `row_idx = tl.program_id(0)`, asks "which row am I?" Each program then advances its `Y`, `X`, and `r` pointers by `row_idx * row_stride` so it operates on its own row and nobody else's. There is no cross-row communication — RMSNorm normalizes each token independently — so the rows are embarrassingly parallel and map cleanly onto the GPU's thousands of resident threads.

![One Triton program per row: tl.program_id(0) maps each row to its own program, and the whole reduction happens in registers; only y and the tiny r leave the chip.](/imgs/blogs/unsloth-triton-kernel-fusion-4.webp)

**Columns are a vector in registers.** `col_offsets = tl.arange(0, BLOCK_SIZE)` builds a vector of column indices `[0, 1, ..., BLOCK_SIZE-1]`, and `mask = col_offsets < n_cols` guards the tail when the hidden dimension is not a power of two. `BLOCK_SIZE` and `num_warps` come from `calculate_settings(n_cols)` — a helper that rounds `n_cols` up to the next power of two for `BLOCK_SIZE` and picks a warp count to match. The single `tl.load(X + col_offsets, mask=mask, other=0)` pulls the *entire row* into an on-chip tensor in one shot, immediately casting to `tl.float32`. The masked-out tail loads as zero so it contributes nothing to the sum.

**The whole reduction stays in float32 registers.** This is the heart of it:

```python
row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
inv_var = tl.math.rsqrt(row_var + eps_f32)
```

`X_row * X_row` is the square (PyTorch's kernel 1) — but the squared values are a register tensor, never written anywhere. `tl.sum(..., axis=0)` is the mean's numerator (kernel 2) computed as an in-register reduction across the block. The divide by `n_cols`, the `+ eps`, and the `rsqrt` (kernels 2 and 3) are scalar register ops. Doing the accumulation in `float32` even when the activation is bf16 matters for numerical stability — squaring bf16 values and summing thousands of them in bf16 would lose precision badly — and it costs nothing extra because registers are wide. None of `squared`, `mean_sq`, or `inv_rms` exists as an HBM tensor. They are transient register values, born and consumed inside the program.

<figure class="blog-anim">
<svg viewBox="0 0 720 260" role="img" aria-label="Inside one program a highlight sweeps across the columns of a row while the sum-of-squares accumulator fills, then rsqrt produces the inverse RMS" style="width:100%;height:auto;max-width:820px">
<title>The in-register reduction: sweep columns, accumulate sum of squares, rsqrt</title>
<style>
.rs-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.rs-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.rs-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.rs-win{fill:none;stroke:var(--accent,#6366f1);stroke-width:3}
.rs-bar{fill:var(--accent,#6366f1)}
.rs-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
@keyframes rs-sweep{0%{transform:translateX(0)}90%{transform:translateX(560px)}100%{transform:translateX(560px)}}
@keyframes rs-grow{0%{width:0}90%{width:600px}100%{width:600px}}
.rs-w{animation:rs-sweep 8s linear infinite}
.rs-g{animation:rs-grow 8s linear infinite}
@media (prefers-reduced-motion:reduce){.rs-w{animation:none;transform:translateX(560px)}.rs-g{animation:none;width:600px}}
</style>
<text class="rs-lbl" x="360" y="28">X_row: col_offsets 0 .. n_cols (one row, in registers)</text>
<rect class="rs-cell" x="40"  y="50" width="60" height="56" rx="6"/>
<rect class="rs-cell" x="120" y="50" width="60" height="56" rx="6"/>
<rect class="rs-cell" x="200" y="50" width="60" height="56" rx="6"/>
<rect class="rs-cell" x="280" y="50" width="60" height="56" rx="6"/>
<rect class="rs-cell" x="360" y="50" width="60" height="56" rx="6"/>
<rect class="rs-cell" x="440" y="50" width="60" height="56" rx="6"/>
<rect class="rs-cell" x="520" y="50" width="60" height="56" rx="6"/>
<rect class="rs-cell" x="600" y="50" width="60" height="56" rx="6"/>
<rect class="rs-win rs-w" x="36" y="46" width="68" height="64" rx="8"/>
<text class="rs-sub" x="360" y="150">accumulate  row_var += x*x   (float32)</text>
<rect class="rs-track" x="60" y="166" width="600" height="34" rx="8"/>
<rect class="rs-bar rs-g" x="60" y="166" width="0" height="34" rx="8"/>
<text class="rs-sub" x="360" y="232">then inv_var = rsqrt(row_var + eps); store r (one float)</text>
</svg>
<figcaption>Inside a single program: a highlight sweeps the row's columns, the float32 sum-of-squares accumulator fills, and rsqrt yields the inverse RMS, all without leaving the chip.</figcaption>
</figure>

**Only the result and one tiny scalar leave.** Two stores happen. `tl.store(Y + col_offsets, output, mask=mask)` writes the final normalized, scaled row to HBM — this is the one output anyone wants. And `tl.store(r, inv_var)` writes a *single float32* per row: the inverse RMS. That `r` tensor is `n_rows` floats total, kept around solely so the backward pass can reuse it instead of recomputing the variance. (The whole backward story — reusing `r`, writing `dX` in place over `dY` — is the subject of the [manual-backprop post](/blog/machine-learning/open-source-library/unsloth-manual-backprop); here it is enough to note that the forward saves exactly one float per row.)

Now the wrapper that drives it, also verbatim, eliding the backward for brevity:

```python
class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma = False):
        shape = X.shape; dim = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = X.device)
        r = torch.empty(n_rows, dtype = torch.float32, device = X.device)  # 1 float/row
        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        with torch_gpu_device(X.device):
            fx[(n_rows,)](Y, Y.stride(0), X, X.stride(0), W, W.stride(0),
                          r, r.stride(0), n_cols, eps,
                          BLOCK_SIZE = BLOCK_SIZE, num_warps = num_warps)
        ctx.eps = eps; ctx.BLOCK_SIZE = BLOCK_SIZE; ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)   # save only X, W, and the tiny r
        return Y.view(*shape)

    # backward(...) reuses r and writes dX in place — see the manual-backprop post

@torch.compiler.disable
def fast_rms_layernorm(layernorm, X, gemma = False):
    W   = layernorm.weight
    eps = layernorm.variance_epsilon if hasattr(layernorm, "variance_epsilon") else layernorm.eps
    return Fast_RMS_Layernorm.apply(X, W, eps, gemma)
```

The wrapper flattens the input to `(n_rows, n_cols)`, allocates exactly two outputs (`Y` the result and `r` the per-row inverse-RMS), computes `BLOCK_SIZE`/`num_warps`, and launches the kernel on the grid `(n_rows,)`. That single launch replaces all five PyTorch launches. The `gemma` flag swaps in a variant for Gemma's slightly different normalization, but the structure is identical. Note `save_for_backward(X, W, r)` — it stashes the input, the weight, and the tiny `r`, *not* the normalized activations, which is itself a memory win for the backward pass.

So the scorecard for RMSNorm:

| | Naive PyTorch eager | Unsloth fused Triton |
| --- | --- | --- |
| Kernel launches | 5 | 1 |
| HBM passes over full activation | ~6 (read x, write/read sq, read x, write/read normed, write y) | 2 (read x, write y) |
| Intermediate HBM tensors | 4 (`sq`, `ms`, `ir`, `normed`) | 0 |
| Extra saved for backward | none (or all activations under autograd) | 1 float per row (`r`) |
| Reduction precision | bf16 unless promoted | float32, always |
| Output | identical | identical |

Same answer, a third of the bytes, a fifth of the launches.

## 5. SwiGLU fused: SiLU and the gate multiply in one pass

The MLP block in Llama-family models is **SwiGLU**: a gated activation where the hidden projection splits into two halves and one half gates the other through a SiLU nonlinearity. Written out, with `e` the gate projection and `g` the up projection:

$$\text{SwiGLU}(x) = \big(\text{SiLU}(e) \odot g\big), \quad \text{SiLU}(e) = e \cdot \sigma(e)$$

In naive PyTorch that is, again, several kernels: a `sigmoid`, a multiply to form SiLU, and a second multiply by the gate — each one reading and writing a full hidden-sized activation. The `e` and `g` tensors for a SwiGLU MLP are large (the intermediate dimension is often 2.5–4× the hidden size), so every pass is expensive. Here is Unsloth's fused forward kernel from `unsloth/kernels/swiglu.py`:

```python
BLOCK_SIZE = 1024

@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr, LONG_INDEXING: tl.constexpr):
    # block_idx -> the slice of the flattened tensors this program owns
    ...
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_row = e_row * tl.sigmoid(e_row)        # SiLU(e) = e * sigmoid(e)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row                    # SwiGLU = SiLU(e) * g, fused, one pass
    tl.store(h + offsets, h_row, mask=mask)
```

The shape of the win is the same as RMSNorm, applied to a different op.

![Fused SwiGLU: the _fg_kernel reads e and g, computes SiLU(e) and multiplies by the gate in one pass, and writes only h; the SiLU intermediate never touches HBM.](/imgs/blogs/unsloth-triton-kernel-fusion-5.webp)

Walk it:

- **Two reads, one write.** The kernel loads a slice of `e` (cast to float32 for the sigmoid) and the matching slice of `g`. It computes `f_row = e_row * sigmoid(e_row)` — that is SiLU — entirely in registers, then casts back to the gate's dtype, multiplies by `g_row`, and stores `h`. The SiLU activation `f` is *never materialized in HBM*. A naive implementation would write SiLU(e) to a full intermediate tensor and read it back for the gate multiply; here it lives and dies in registers, the same trick as RMSNorm's `squared`.
- **Elementwise, so it tiles by flat index.** Unlike RMSNorm there is no per-row reduction — SwiGLU is pure elementwise — so the kernel does not need one-program-per-row. It flattens `e`, `g`, `h` and tiles them into `BLOCK_SIZE`-element chunks, one program per chunk, with the usual `mask` for the tail.
- **`LONG_INDEXING` is the large-tensor guard.** This is the detail worth pausing on. The flattened element offset into `e`/`g`/`h` is `block_idx * BLOCK_SIZE + arange(0, BLOCK_SIZE)`. For a big training batch the total element count `n_elements` can exceed $2^{31} \approx 2.1$ billion — e.g. a long-sequence batch through a wide MLP intermediate. A 32-bit integer offset would silently overflow and corrupt memory access. The `LONG_INDEXING: tl.constexpr` flag, set by the host wrapper when `n_elements > 2**31`, switches the kernel to compute offsets in `int64`. It is a compile-time constant so there is no per-element branch — Triton specializes the kernel for the 32-bit and 64-bit cases separately, and you pay the wider arithmetic only when you actually need it. This is the kind of correctness detail that separates a kernel that works on toy inputs from one that survives real training runs.

The backward kernel, `_DWf_DW_dfg_kernel`, applies the same fusion philosophy to the three derivatives SwiGLU needs (`h`, `df`, `de`), overwriting the `e`/`g`/`DW` buffers in place so no new tensors are allocated — again, the territory of the [manual-backprop post](/blog/machine-learning/open-source-library/unsloth-manual-backprop). And the forward `_fg_kernel` is exactly the `_forward_function` that Unsloth's fused LoRA MLP (`LoRA_MLP.apply(..., swiglu_fg_kernel, ...)`) calls to compute `h = SwiGLU(X@gate, X@up)` between the LoRA-augmented projections. The fusion is not an isolated micro-optimization; it slots directly into the autograd function that runs the whole MLP.

## 6. Why fusion wins twice: launches and bandwidth

Fusion pays off on two independent axes, and it is worth separating them because they scale differently.

**Axis one: fewer launches (latency).** Each saved kernel launch saves a few microseconds of fixed host-side overhead. RMSNorm goes from 5 launches to 1 — four launches saved per normalization. A Llama-style model has two RMSNorms per layer (input and post-attention) plus a final norm; across, say, 32 layers that is ~65 normalizations × 4 saved launches = ~260 fewer launches per forward pass, and the same again on the backward. At a few microseconds each, that is on the order of a millisecond of pure launch overhead removed per step — small per step, but it compounds over hundreds of thousands of steps, and it is *latency you were paying for nothing*. For small batches, where each kernel's actual work is tiny, launch overhead can be the dominant cost, and fusion's launch savings matter even more than its bandwidth savings.

**Axis two: fewer HBM round-trips (bandwidth).** This is the bigger lever for realistic batch sizes, and it is worth doing the arithmetic.

![Bytes moved per RMSNorm row: the unfused path is ~48 KB and 5 launches; the fused kernel is ~16 KB and 1 launch — roughly a 3x cut.](/imgs/blogs/unsloth-triton-kernel-fusion-6.webp)

Take one row of `n_cols = 4096` bf16 elements — 8 KB. Trace the HBM traffic the naive RMSNorm generates for that row:

1. `x.pow(2)`: read `x` (8 KB) + write `sq` (8 KB) = 16 KB
2. `.mean(-1)`: read `sq` (8 KB) + write `ms` (tiny) ≈ 8 KB
3. `rsqrt`: read/write `ms`/`ir` (tiny) ≈ 0 KB
4. `x * inv_rms`: read `x` again (8 KB) + write `normed` (8 KB) = 16 KB
5. `normed * w`: read `normed` (8 KB) + write `y` (8 KB) = 16 KB

That is roughly **48 KB of HBM traffic** for one row (ignoring the small-tensor noise). The fused kernel does:

1. read `x` (8 KB)
2. write `y` (8 KB)
3. write `r` (4 bytes)

≈ **16 KB**. A clean **3× reduction** in bytes moved, which — because RMSNorm is memory-bound at arithmetic intensity ~1 — translates almost directly into a 3× speedup *for that op*. Scale the 8 KB per row up to a full `(4, 2048, 4096)` activation and you are moving ~96 MB through HBM unfused versus ~32 MB fused, per RMSNorm, per pass. SwiGLU shows the same pattern: unfused materializes the SiLU intermediate (read `e`, write `f`, read `f`, read `g`, write `h`), fused reads `e` and `g` once and writes `h` once.

The two axes interact. On large batches, bandwidth dominates and the 3× byte reduction is the story. On small batches — common in interactive fine-tuning and the GRPO rollouts Unsloth optimizes heavily — launch latency dominates and the 5-to-1 launch reduction is the story. Fusion happens to win on whichever axis is binding, which is why the speedup holds across batch regimes. And critically, this is "3x faster training and 30% less VRAM via new Triton kernels" as the README phrases it — *both* axes of the kernel-fusion win, not a precision tradeoff.

## 7. The `@torch.compiler.disable` detail

Look back at the RMSNorm wrapper and you will see the public entry point is decorated:

```python
@torch.compiler.disable
def fast_rms_layernorm(layernorm, X, gemma = False):
    ...
    return Fast_RMS_Layernorm.apply(X, W, eps, gemma)
```

`@torch.compiler.disable` (and `@torch._disable_dynamo`, used similarly on the RoPE kernels) tells PyTorch's compiler stack to treat this function as an opaque boundary: do not trace into it, do not try to fuse it with surrounding graph, do not recompile it. At first glance that looks backwards — isn't `torch.compile` *also* a fusion engine? Why opt out of it?

The reasoning is sound. First, **the kernel is already optimally fused by hand.** `torch.compile`'s value is automatically fusing the elementwise tail that eager mode leaves split; Unsloth has already done that fusion manually and provably (one launch, two HBM passes, float32 reduction). Letting the compiler trace through a hand-written Triton kernel and a custom `autograd.Function` adds nothing — there is nothing left to fuse — and it risks the compiler making a *worse* decision than the hand-tuned one.

Second, **`torch.compile` interacts badly with custom `autograd.Function`s and Triton kernels at the tracing boundary.** Dynamo has to special-case custom autograd functions, and graph breaks around them are common; a custom Triton kernel inside a compiled region can trigger recompilations, guard failures, or simply opaque graph breaks that you would rather declare explicitly. By marking the boundary with `@torch.compiler.disable`, Unsloth makes the contract unambiguous: "this is a leaf, compiled code calls it like any other op, do not look inside." The compiler optimizes the matmuls and the graph around the norm; the norm itself is the hand-fused kernel, full stop.

Third, **determinism and numerics.** Unsloth's selling point is bit-for-bit-faithful, no-approximation kernels. A hand-written kernel whose float32 reduction and dtype casts are pinned down is a fixed, auditable quantity. Handing it to a compiler that may reorder operations, change accumulation order, or fuse across the boundary introduces variability that is hard to reason about. Disabling the compiler on these kernels keeps the numerics exactly where Unsloth put them.

The general principle: `torch.compile` is a fantastic tool for the code you *haven't* hand-optimized. For the kernels that are the entire point of your library — already maximally fused, numerically pinned, wrapped in custom autograd — the right move is to fence them off and let the compiler handle everything else.

## 8. Numbers, case studies, and when to reach for this

The fused kernels are not a micro-benchmark curiosity; they are why Unsloth's published numbers exist. Let me ground the discussion in concrete situations where the fusion mechanism is the visible cause of the result.

**Case 1: The "3x faster / 30% less VRAM" headline.** Unsloth's 2026 README attributes a specific tier of its speedup to "new Triton kernels and padding-free packing." Decompose that. The 3× speed is the launch-and-bandwidth win we traced: every normalization and activation in the model goes from many bandwidth-bound launches to one, and on the memory-bound ops that is a near-linear time reduction. The 30% less VRAM is partly the absence of the intermediate tensors (`squared`, `normed`, the SiLU activation) that the naive path allocates in HBM, and partly the tiny `r`-only backward save. The kernels are the engine; padding-free packing just makes sure the kernels aren't wasting that bandwidth on pad tokens.

**Case 2: bf16 reduction precision, for free.** A subtle but real benefit: the fused RMSNorm always accumulates the sum-of-squares in float32 (`X_row ... .to(tl.float32)` then `tl.sum`). A naive bf16 `x.pow(2).mean(-1)` accumulates in bf16 unless you remember to upcast, and summing 4096 squared bf16 values in bf16 accumulates rounding error that can visibly perturb the normalization. Unsloth's kernel gives you the numerically-correct float32 reduction *and* the speed, because the upcast is a register operation that costs nothing. The fused path is not just faster; for the reduction it is more accurate than the lazy eager version.

**Case 3: small-batch GRPO and the launch wall.** Reinforcement-learning fine-tuning (GRPO and friends) runs many short generations with small effective batches. Here the per-kernel *work* is small, so launch overhead dominates — exactly the regime where collapsing 5 launches into 1 matters most. Unsloth advertises "80% less VRAM for GRPO" and a 2× speedup on the gpt-oss-20B GRPO path; the launch-count reduction from fusion is a direct contributor, because in small-batch regimes you are paying launch latency, not bandwidth.

**Case 4: the LONG_INDEXING near-miss class of bug.** Anyone who has written GPU kernels has hit, or narrowly avoided, the int32-offset overflow. A SwiGLU MLP intermediate on a long-context batch can easily exceed $2^{31}$ elements. Without the `LONG_INDEXING` guard, the kernel would compute a wrapped negative offset, read or write the wrong global-memory address, and produce either garbage gradients or an illegal-memory-access crash — and it would only trigger on large inputs, making it a nightmare to reproduce. Unsloth's compile-time `int64` switch is the boring, correct fix: branch once at compile time on tensor size, pay the wider arithmetic only when the tensor is actually huge.

**Case 5: where fusion does *not* help — the matmuls.** It is worth stating the limit. Fusion's win is on memory-bound ops. The dense projections (Q/K/V, gate/up/down) are compute-bound matmuls; you do not hand-fuse those into a single elementwise-style Triton kernel, because they are already keeping the tensor cores busy. Unsloth's matmul story is different — it is about quantized GEMMs and the [LoRA matmul path](/blog/machine-learning/open-source-library/unsloth-manual-backprop), not RMSNorm-style fusion. Reaching for elementwise fusion on a compute-bound op buys nothing; know which side of the ridge point you are on before you optimize.

**Case 6: attention is delegated, not fused here.** Another limit worth naming: Unsloth does *not* write a custom fused attention kernel. Attention is handed to Flash Attention / xformers, which are already the state of the art at keeping the softmax and the score matrix on-chip. Unsloth's contribution around attention is feeding it correctly-laid-out tensors and a fused RoPE (the same one-kernel-per-row philosophy, applied to the rotary embedding). Knowing what *not* to rewrite is as much a part of the design as the kernels they do write.

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="Unfused RMSNorm bounces a row of activations between HBM and the ALUs five times, while the fused kernel makes one round-trip" style="width:100%;height:auto;max-width:820px">
<title>Unfused vs fused: HBM round-trips for one RMSNorm row</title>
<style>
.uf-band{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.uf-hbm{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.uf-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.uf-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.uf-pkt{fill:var(--accent,#6366f1)}
.uf-fpkt{fill:#22a06b}
@keyframes uf-bounce{0%{transform:translateY(0)}10%{transform:translateY(120px)}20%{transform:translateY(0)}30%{transform:translateY(120px)}40%{transform:translateY(0)}50%{transform:translateY(120px)}60%{transform:translateY(0)}70%{transform:translateY(120px)}80%{transform:translateY(0)}90%{transform:translateY(120px)}100%{transform:translateY(0)}}
@keyframes uf-once{0%{transform:translateY(0)}45%{transform:translateY(120px)}55%{transform:translateY(120px)}100%{transform:translateY(0)}}
.uf-b{animation:uf-bounce 9s ease-in-out infinite}
.uf-o{animation:uf-once 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.uf-b,.uf-o{animation:none}}
</style>
<text class="uf-lbl" x="180" y="28">Unfused: 5 round-trips</text>
<rect class="uf-hbm" x="60" y="50" width="240" height="40" rx="6"/>
<text class="uf-sub" x="180" y="75">HBM</text>
<rect class="uf-band" x="60" y="210" width="240" height="40" rx="6"/>
<text class="uf-sub" x="180" y="235">ALUs (square,mean,rsqrt,scale)</text>
<rect class="uf-pkt uf-b" x="160" y="96" width="40" height="40" rx="6"/>
<text class="uf-lbl" x="540" y="28">Fused: 1 round-trip</text>
<rect class="uf-hbm" x="420" y="50" width="240" height="40" rx="6"/>
<text class="uf-sub" x="540" y="75">HBM</text>
<rect class="uf-band" x="420" y="210" width="240" height="40" rx="6"/>
<text class="uf-sub" x="540" y="235">one kernel (registers/SRAM)</text>
<rect class="uf-fpkt uf-o" x="520" y="96" width="40" height="40" rx="6"/>
<text class="uf-sub" x="180" y="290">each trip writes a tmp tensor</text>
<text class="uf-sub" x="540" y="290">intermediates never leave the chip</text>
</svg>
<figcaption>One RMSNorm row: the unfused path bounces between HBM and the ALUs once per op (five times); the fused kernel reads once, computes on-chip, and writes once.</figcaption>
</figure>

**Case 7: why you cannot just "use bigger batches" to fix the naive version.** A tempting counterargument: if the problem is launch overhead, won't a big enough batch amortize it? Partly — bigger batches do amortize launch latency. But they do *not* fix the bandwidth problem; in fact they make it worse in absolute terms, because the intermediate tensors scale with batch size. A bigger batch means bigger `squared` and `normed` tensors round-tripping HBM, more bytes moved, more VRAM consumed by intermediates you never wanted. The fused kernel's advantage is structural — it eliminates the intermediates entirely — so it wins at every batch size, just for different reasons (latency at small, bandwidth at large).

**Case 8: the debugging payoff of "no approximations."** Because the kernels are exact rewrites, a model fine-tuned with Unsloth produces the same loss curve as the equivalent vanilla-PyTorch run, modulo the float32-vs-bf16 reduction precision which only makes Unsloth *more* accurate. When a fine-tune misbehaves, you do not have to wonder whether a fused kernel silently changed the math. That is a real operational benefit: the fast path and the reference path agree, so you can debug the model, not the kernel.

### When to reach for hand-fused Triton kernels

Reach for hand fusion when **the op is memory-bound** (low arithmetic intensity — normalizations, activations, elementwise chains, small reductions), when **the same intermediate is produced and immediately consumed** (so it can live in registers instead of HBM), and when **you are willing to pin the numerics** (do the reduction in float32, control the casts) and own the backward pass. RMSNorm and SwiGLU are the canonical examples: cheap math, expensive memory, an obvious intermediate to keep on-chip.

Do **not** reach for it when the op is already compute-bound (dense matmuls — let cuBLAS/the tensor cores do their job), when a battle-tested kernel already exists (attention — use Flash Attention), or when the code is not on the hot path (fusing a once-per-step scalar op buys nothing measurable). And before reaching for a hand-written kernel at all, check whether `torch.compile` already fuses the pattern adequately — for code you have *not* deliberately hand-tuned, it often does, and it is far less work. Unsloth fences `torch.compile` off its kernels precisely because those kernels are the ones it *has* hand-tuned to the metal; everything else, it is happy to let the compiler fuse.

The deeper lesson outlives any single library: on modern accelerators, the first question about any operation is not "how many FLOPs?" but "how many bytes cross HBM, and how many times?" Get the arithmetic intensity right, keep the transient values on-chip, and collapse the launches — and you get a 3× speedup with zero change to the answer. That is the whole trick, and Unsloth's RMSNorm and SwiGLU kernels are two of the cleanest examples of it you will find in production code.
