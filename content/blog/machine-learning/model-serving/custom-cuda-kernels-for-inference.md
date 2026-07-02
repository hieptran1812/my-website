---
title: "Custom CUDA Kernels for Inference: When to Drop Below PyTorch Ops"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A principal engineer's guide to hand-written and library GPU kernels for LLM serving — the memory-wall math that motivates them, a real Triton fused kernel, how vLLM and SGLang wire them in, and the before-after numbers on H100 and A100."
tags:
  [
    "model-serving",
    "inference",
    "cuda",
    "triton",
    "flashattention",
    "gpu-kernels",
    "quantization",
    "kernel-fusion",
    "marlin",
    "ml-infrastructure",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/custom-cuda-kernels-for-inference-1.webp"
---

At 2 a.m. the pager went off for a decode-latency SLO breach. The service — a Llama-3-8B chat endpoint on a single A100 — was serving the same request rate it had handled for weeks, but per-token latency (TPOT) had crept from 22 ms to 41 ms after we swapped in a new attention implementation "to be safe." GPU utilization sat at a comfortable-looking 60%. The temptation was to blame the batch scheduler or the network. But when I attached Nsight Systems and looked at the kernel timeline, the truth was uglier and more mundane: the GPU spent more time *waiting between kernels* than computing inside them. Each decode step launched roughly three hundred tiny CUDA kernels — a norm here, a RoPE rotation there, an elementwise multiply for the gate — and every one of them read its inputs from high-bandwidth memory (HBM), did a trickle of arithmetic, and wrote the result straight back to HBM. The 312 teraFLOPS of tensor-core compute on that card were almost entirely idle. We were not compute-starved. We were *memory-starved*, and we were paying kernel-launch tax on top of it.

This is the situation that custom GPU kernels exist to fix. The default PyTorch execution model — one CUDA kernel per operator, each round-tripping through HBM — is convenient, portable, and, for inference at the token-by-token scale, frequently the single biggest thing standing between you and your latency budget. The figure below shows the canonical example: the attention operation, written as three separate kernels that materialize an $S \times S$ score matrix in HBM, versus FlashAttention, which fuses the entire computation into one kernel that never writes that matrix out at all.

![Before-after comparison of unfused three-kernel attention that writes the S by S score matrix to HBM versus FlashAttention as one fused kernel that keeps tiles in SRAM](/imgs/blogs/custom-cuda-kernels-for-inference-1.webp)

By the end of this post you will be able to: read a roofline plot and predict whether an operator is memory- or compute-bound; explain the I/O math that makes FlashAttention faster without changing a single FLOP; write a real fused kernel in Triton and launch it; wrap it as a PyTorch custom op so `torch.compile` and vLLM can use it safely; reach for the right quantized-GEMM kernel (Marlin, W4A16, FP8) for your hardware; and — most importantly — decide *when the effort is worth it and when `torch.compile` or an off-the-shelf library kernel is already enough*. This is the sibling of [kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile), which covers the *compiler-generated* path; this post is about the kernels you (or a library author) write by hand.

Every technique here is a trade on the serving SLO triangle — **latency ↔ throughput ↔ cost**. A faster kernel can buy you lower TPOT, more tokens per second per GPU, or fewer GPUs for the same load. It costs engineering time, portability, and maintenance. We will be explicit about both sides.

## The memory wall: your GPU is starving, not thinking

Start with the one number that governs everything: the ratio of a chip's compute throughput to its memory bandwidth. An NVIDIA A100 80GB SXM does about 312 teraFLOPS of dense FP16 tensor-core math and moves about 2.0 TB/s across its HBM2e. An H100 SXM5 does roughly 990 TFLOPS dense BF16 and 3.35 TB/s of HBM3. Divide the two and you get the **ridge point** of the roofline — the arithmetic intensity at which a kernel transitions from memory-bound to compute-bound:

$$
\text{ridge}_{\text{A100}} = \frac{312 \times 10^{12} \text{ FLOP/s}}{2.0 \times 10^{12} \text{ B/s}} \approx 156 \text{ FLOP/byte}, \qquad
\text{ridge}_{\text{H100}} = \frac{990 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ FLOP/byte}.
$$

**Arithmetic intensity** (AI) is FLOPs performed divided by bytes moved to and from HBM. If your operator's AI is *below* the ridge point, you are memory-bound: the tensor cores finish their work and sit idle waiting for the next byte to arrive, and your kernel's wall-clock time is $\text{bytes} / \text{bandwidth}$, full stop. If AI is above the ridge, you are compute-bound and time is $\text{FLOPs} / \text{peak}$. The roofline model is just those two lines; the ridge is where they meet.

Now look at what LLM inference actually does. Almost all of decode — the autoregressive, one-token-at-a-time phase that dominates serving cost — sits far below the ridge. The figure below tabulates the arithmetic intensity of the operators in a transformer decoder step, and the pattern is stark.

![Matrix comparing arithmetic intensity, roofline bound, HBM traffic, and fusion payoff across decode GEMV, prefill GEMM, attention decode, attention prefill, RMSNorm, RoPE, and SwiGLU](/imgs/blogs/custom-cuda-kernels-for-inference-2.webp)

Consider the projection matmuls at batch size 1. A decode step multiplies a weight matrix $W \in \mathbb{R}^{n \times k}$ by a single activation vector $x \in \mathbb{R}^{k}$. That is ${2nk}$ FLOPs and it reads ${2nk}$ bytes of FP16 weights. The arithmetic intensity is roughly ${2nk / 2nk = 1}$ FLOP/byte — a hundred and fifty times below the A100 ridge. This is not a matmul in any performance-relevant sense; it is a memory copy with a multiply-accumulate stapled on. The same is true of the elementwise operators: RMSNorm, RoPE, the SwiGLU gate, residual adds. They read a tensor, touch each element once or twice, and write it back. Their AI is well under 1.

#### Worked example: the single-stream decode floor on A100 vs H100

Llama-3-8B has about 8 billion parameters, which in FP16 is roughly 16 GB. At batch size 1, generating one token requires reading *every weight once* (plus the KV cache, which we will ignore for the floor). The minimum possible time for one decode step is therefore bounded by bandwidth:

$$
t_{\text{min}}^{\text{A100}} = \frac{16 \times 10^{9} \text{ B}}{2.0 \times 10^{12} \text{ B/s}} \approx 8.0 \text{ ms} \Rightarrow \lesssim 125 \text{ tok/s}, \qquad
t_{\text{min}}^{\text{H100}} = \frac{16 \times 10^{9}}{3.35 \times 10^{12}} \approx 4.8 \text{ ms} \Rightarrow \lesssim 208 \text{ tok/s}.
$$

No kernel, however clever, beats this floor at batch 1 — it is a property of the weights and the bus, not the code. This is *why* LLM serving batches aggressively: batching amortizes the weight read across many tokens, raising arithmetic intensity toward the ridge. It is also why quantization is so effective for decode — halving the weight bytes with INT4 nearly halves the floor. The full argument for why decode is memory-bound and prefill is compute-bound is developed in [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different); here the point is narrower. When you are memory-bound, the only lever that matters is **bytes moved**. And the fastest way to move fewer bytes is to stop writing intermediate results to HBM — which is exactly what a fused kernel does.

### Batching moves you up the roofline

The reason batch size dominates every serving conversation is that it is the arithmetic-intensity dial. When you process $B$ tokens through the same weight matrix, you read the weights *once* (${2nk}$ bytes) and do $B$ times the FLOPs (${2nkB}$). Arithmetic intensity is therefore ${2nkB / 2nk = B}$ FLOP/byte — it scales linearly with batch. On A100 the ridge is ~156, so the projection GEMMs stay memory-bound until roughly $B \approx 156$, and only above that does the matmul become compute-bound and start using the tensor cores at their rated throughput. This single fact explains a huge amount of serving behavior: at batch 1 you are 150× below the ridge and no kernel can help beyond cutting bytes; at batch 256 you are compute-bound and a *better GEMM* kernel (FP8, better tiling) starts to matter. The whole discipline of [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) exists to keep $B$ as high as latency allows, precisely so the expensive silicon stops idling.

There is a second bandwidth consumer that batching does *not* amortize: the KV cache. Every decode step, attention must read the entire key-value cache for each active sequence, and unlike weights, the KV cache is per-sequence — it scales with batch. For Llama-3-8B with grouped-query attention (8 KV heads, head dimension 128, 32 layers, FP16), one token of context costs $2 \times 8 \times 128 \times 2 \times 32 = 131{,}072$ bytes ≈ 131 KB across all layers.

#### Worked example: when the KV cache read overtakes the weight read

At sequence length 4,096, one sequence's KV cache is $4096 \times 131 \text{ KB} \approx 537$ MB. Reading it once per decode step costs $537 \times 10^6 / 2.0 \times 10^{12} \approx 0.27$ ms on A100 — small next to the 8 ms weight read. But KV read scales with batch while the weight read does not. At batch $B$, total decode-step HBM traffic is roughly $16 \text{ GB} + B \times 0.537 \text{ GB}$. The KV term equals the weight term at $B \approx 30$; past that, the KV cache — not the weights — is what your bandwidth is spent on, and *KV-cache kernels* (paged attention, FP8 KV, prefix sharing) become the lever rather than weight quantization. This crossover is why long-context, high-batch serving invests so heavily in KV-cache-side kernels; it is developed further in [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management). The lesson for kernel work: *profile which read dominates at your batch and context, then optimize that read* — do not quantize weights when your bytes are going to KV.

### Attention and the $O(N^2)$ HBM tax

Attention deserves its own paragraph because it is where the memory wall does the most damage and where fused kernels win biggest. The naive computation is three steps: $S = QK^\top$, then $P = \text{softmax}(S)$, then $O = PV$. The score matrix $S$ is $N \times N$ for a sequence of length $N$. Written as separate kernels, the $QK^\top$ kernel *writes* $S$ to HBM, the softmax kernel *reads and writes* it, and the $PV$ kernel *reads* it again. That is the $O(N^2)$ term that dominates: for $N = 8192$ and a single FP16 head, $S$ is $8192^2 \times 2 \approx 134$ MB, and it gets moved across the bus several times per head, per layer, per step.

The FlashAttention paper (Dao et al., 2022) states the bound precisely: standard attention performs $\Theta(Nd + N^2)$ HBM accesses, while FlashAttention performs $\Theta(N^2 d^2 M^{-1})$, where $d$ is the head dimension and $M$ is the on-chip SRAM size. Because $d^2$ (for $d = 128$, that is 16,384 elements) is smaller than $M$ (an A100 SM has up to 192 KB of shared memory, tens of thousands of FP16 elements), FlashAttention moves dramatically fewer bytes — and, crucially, it *never materializes the $N \times N$ matrix in HBM at all*. It tiles $Q$, $K$, $V$ into blocks that fit in SRAM, computes partial scores there, and uses an **online softmax** (a running max and running sum) to combine block results without ever seeing the full row at once.

#### Worked example: FlashAttention HBM traffic at 8k context

For one head at $N = 8192$, $d = 128$, FP16: the naive path writes $S$ ($\approx 134$ MB), the softmax reads and writes it again ($\approx 268$ MB), and $PV$ reads it once more ($\approx 134$ MB) — on the order of **500+ MB of HBM traffic per head just for the score matrix**, before counting $Q$, $K$, $V$, $O$. FlashAttention reads $Q$, $K$, $V$ and writes $O$ once each: $4 \times (8192 \times 128 \times 2) \approx 8.4$ MB per head. Same FLOPs, roughly two orders of magnitude less HBM traffic on the dominant term. That is the entire source of the 2–4× speedup — not faster math, *less memory movement*. This is the same memory-wall argument developed at the hardware level in [kernel fusion and FlashAttention: beating the memory wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall).

### The online-softmax trick that makes fusion possible

The obstacle to fusing attention is softmax's global dependency: to normalize row $i$ you need the sum of $e^{s_{ij}}$ over *all* $j$, which seems to require the whole row in memory at once — the very thing we are trying to avoid. FlashAttention sidesteps this with an online (streaming) softmax that maintains two running statistics per query row as it walks the key blocks: the running max $m$ and the running normalizer $\ell$. When a new block of scores $x$ arrives, the update is

$$
m^{\text{new}} = \max(m,\ \text{rowmax}(x)), \qquad
\ell^{\text{new}} = e^{\,m - m^{\text{new}}}\,\ell + \sum_j e^{\,x_j - m^{\text{new}}},
$$

and the output accumulator $O$ is rescaled by the same correction factor $e^{\,m - m^{\text{new}}}$ before the current block's contribution $\big(\sum_j e^{x_j - m^{\text{new}}} v_j\big)$ is added. When the last block is processed, a single division by $\ell$ finishes the softmax. The correction factor is the whole trick: it retroactively fixes up the earlier partial sums whenever a later block reveals a larger maximum, so the result is bit-for-bit the same as a full-row softmax while only ever holding one block in SRAM. This is what lets the kernel loop over $K$ and $V$ in tiles — reading each exactly once — instead of materializing the full score row. Causal masking drops out naturally: blocks entirely above the diagonal are skipped, which is why causal attention is roughly half the work of full attention in these kernels.

### Calling FlashAttention in practice

You almost never write an attention kernel; you *call* one. Since PyTorch 2.0, the fused path is one function — `F.scaled_dot_product_attention` (SDPA) — which auto-selects a FlashAttention, memory-efficient, or math backend based on dtype, head dimension, and GPU. For serving you usually want to force the fused backend and know it engaged:

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# q, k, v: (batch, n_heads, seq_len, head_dim), fp16/bf16 on CUDA.
# Auto-selects the FlashAttention-2 backend on Ampere+/Hopper when eligible.
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Pin the backend so a shape/dtype quirk cannot silently drop you to the
# O(N^2)-HBM math kernel — the exact regression that pages you at 2 a.m.
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# The reference kernel directly (Dao's flash-attn package). Note the layout:
# (batch, seq_len, n_heads, head_dim) — heads and seq swapped vs SDPA.
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```

For *decode* against a paged KV cache, SDPA and `flash_attn_func` do not fit — the KV is not contiguous. That is FlashInfer's domain: you plan the paged layout once per batch, then run the decode attention over the block table.

```python
import flashinfer

# Plan with the KV page structure (indptr/indices describe the block table).
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
wrapper.plan(
    kv_page_indptr, kv_page_indices, kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim, page_size,
    data_type=torch.float16,
)
o = wrapper.run(q, paged_kv_cache)   # gathers KV blocks, runs online-softmax attention
```

The point of showing all three: the same online-softmax math is reached through three different call sites depending on whether you are doing training-style contiguous attention (SDPA / `flash_attn_func`) or serving-style paged decode (FlashInfer). Picking the wrong one — for instance, forcing contiguous attention and letting the engine copy the paged KV into a dense buffer every step — silently reintroduces the HBM traffic the fused kernel was supposed to avoid.

### Kernel-launch overhead: the second tax

There is a second, independent tax that has nothing to do with bandwidth: **launching a kernel costs time**. Each CUDA kernel launch from the host carries roughly 5–10 µs of overhead (queue submission, driver dispatch, scheduling). That sounds trivial until you count launches.

#### Worked example: the launch-overhead budget of a decode step

Walk one transformer layer of Llama-3-8B at batch 1 and count the kernels a naive eager implementation issues: an input RMSNorm (which itself is a square, a mean-reduce, an rsqrt, and two multiplies — call it 4 kernels), a QKV projection GEMM, a RoPE rotation on Q and K, the attention call, an output projection GEMM, a residual add, a post-attention RMSNorm (4 more), the gate and up projections (2 GEMMs), a SiLU, an elementwise multiply, the down projection GEMM, and a second residual add. That is roughly 18–22 kernels per layer. Across 32 layers plus the final norm and the LM-head projection, a decode step issues on the order of **600–700 kernel launches**. At 7 µs each that is **4.2–4.9 ms of pure launch overhead per token** — which, on an H100 whose *compute* floor for 8B is under 5 ms, can be as large as the compute itself. This is the number that made our 2 a.m. incident: the "safe" attention swap had disabled a fused path and re-exposed dozens of these launches per layer.

Fusing $k$ operators into one kernel removes $k-1$ launches *and* $k-1$ HBM round-trips at once. CUDA graphs attack the launch tax from the other direction — recording the whole launch sequence once and replaying it — and that is the subject of the [torch.compile and CUDA graphs post](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile). The two are complementary: fusion shrinks the number of kernels; CUDA graphs make launching whatever remains nearly free. Custom kernels are how you get fusions the compiler cannot or will not generate.

## The kernel abstraction ladder: how deep do you go?

Before writing anything, decide how far down you need to drop. There is a ladder from "call a PyTorch op" to "hand-schedule tensor-core instructions," and each rung trades away portability and development speed for control over exactly what lands in HBM and how the tensor cores are fed. The figure below lays out the rungs.

![Stack diagram of the kernel abstraction ladder from PyTorch eager ops down through Triton, CUDA C++ and CUTLASS, PTX and SASS assembly, to tensor cores and HBM](/imgs/blogs/custom-cuda-kernels-for-inference-3.webp)

- **PyTorch eager ops** (`torch.matmul`, `F.softmax`). Maximum portability, zero kernel-authoring effort, dispatches to cuBLAS/cuDNN. The cost is one kernel and one HBM round-trip per op, plus a launch each.
- **`torch.compile` / Triton.** The compiler (`torch.compile` with the Inductor backend generates Triton) fuses pointwise chains automatically and can autotune. You can also write Triton by hand — a Python-embedded DSL that compiles to PTX through LLVM. This is the sweet spot for most custom work: you control tiling and what stays in SRAM, but you write Python-like code and get reasonable performance across NVIDIA and AMD.
- **CUDA C++ / CUTLASS.** Hand-written kernels or CUTLASS templates give you warp-level control, `mma` instruction selection, software pipelining, and swizzled shared-memory layouts. This is where the last 20–40% of peak lives for GEMM-heavy work, and where libraries like Marlin and FlashAttention are actually implemented.
- **PTX / SASS.** Inline assembly and hand-tuned instruction scheduling. Per-architecture, brittle, and rarely worth it outside a handful of library hot loops.
- **Tensor cores + HBM.** The hardware you are ultimately trying to keep busy: 3.35 TB/s of bandwidth and ~990 TFLOPS on H100 that a good kernel saturates and a bad one wastes.

The practical guidance: **most inference wins come from the middle two rungs.** Triton for fusing your model-specific elementwise and attention-variant work; CUTLASS/CUDA (usually via a library) for quantized GEMM and attention. You almost never hand-write PTX, and if PyTorch eager is meeting your SLO, you stay there and go do something more valuable.

## Runtime kernel dispatch: how a serving engine picks

You rarely dispatch to one kernel and stop. A production engine chooses, per operator and at runtime, the fastest available kernel given the dtype, the tensor shape, and the GPU architecture — then composes them so the whole layer is as few launches as possible. The figure below sketches that dispatch.

![Graph showing an op request branching through a runtime kernel dispatcher to cuBLAS, a Triton kernel, Marlin W4A16, or an FP8 tensor-core kernel, all converging on one fused launch](/imgs/blogs/custom-cuda-kernels-for-inference-4.webp)

Concretely, vLLM inspects the model's dtype and the current GPU capability at load time. A GPTQ 4-bit checkpoint on an Ampere-or-newer card routes matmuls to the Marlin W4A16 kernel; an FP8 checkpoint on H100 routes to FP8 tensor-core GEMM; plain FP16 falls back to cuBLAS/CUTLASS; attention goes to FlashAttention-2, FlashAttention-3, or FlashInfer depending on what is installed and whether the KV cache is paged. The dispatcher's job is to pick the kernel that minimizes bytes moved for *this* shape on *this* hardware, and the convergence node in the figure is the point: whatever branch is chosen, the goal is one fused launch that touches HBM as little as possible.

This is why "which kernel is fastest" is never a single answer. It is a lookup keyed on (dtype, shape, arch), and the whole reason libraries ship dozens of specialized kernels is to have a good entry for every cell of that table. The [choosing-your-serving-stack](/blog/machine-learning/model-serving/what-is-model-serving) decision — which engine to run at all — sits one level above this; here we are inside the engine, watching it choose.

The dispatch is not purely static, either. Some of it happens at load time (a GPTQ checkpoint on Ampere is bound to Marlin once), but some is per-request: the attention backend may differ between the compute-bound prefill phase (a dense FlashAttention kernel over the whole prompt) and the memory-bound decode phase (a paged-attention kernel gathering one new query against the cached KV). vLLM, for instance, runs chunked prefill through one attention path and single-token decode through another, because the shapes and the roofline position are different. A well-built engine therefore holds not one kernel per operator but a small menu, and the "dispatcher" is really the model-runner code choosing per phase and per shape bucket. When you add a custom kernel, you are adding an entry to that menu — which is why registration and a clean fallback matter as much as the kernel's raw speed: if your kernel is only faster for one shape bucket, the engine must still have a correct, if slower, path for every other cell of the table.

## Writing a fused kernel in Triton

Enough motivation. Here is a real fused kernel. The canonical teaching example — and a genuinely useful one — is a fused softmax. In eager PyTorch, `torch.softmax(x, dim=-1)` on a matrix launches a chain of kernels: a max-reduce, a subtract, an exp, a sum-reduce, and a divide. Each reads the row from HBM and writes it back. Fused, the whole thing is one kernel that loads each row once into SRAM, does all five steps in registers, and writes the result once.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # One program instance handles one row of the input matrix.
    row_idx = tl.program_id(0)
    row_start = in_ptr + row_idx * in_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    ptrs = row_start + col_offsets
    mask = col_offsets < n_cols

    # Load the whole row into SRAM once. Out-of-range lanes read -inf so they
    # vanish under the max and contribute 0 to the exp-sum.
    row = tl.load(ptrs, mask=mask, other=-float("inf"))

    # All of softmax happens in registers/SRAM — no HBM round-trips between steps.
    row_minus_max = row - tl.max(row, axis=0)   # numerically stable
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    # Single write back to HBM.
    out_start = out_ptr + row_idx * out_row_stride
    tl.store(out_start + col_offsets, softmax_out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    # BLOCK_SIZE must cover a full row so the reduction is a single tile.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    # The launch grid: one program per row. grid is a 1-D tuple here.
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

Two things are worth internalizing. First, the **launch grid** `(n_rows,)` says "instantiate this kernel `n_rows` times, and inside each instance `tl.program_id(0)` tells me which row I own." Triton programs are the unit of parallelism; the runtime schedules them across the GPU's streaming multiprocessors. Second, everything between the single `tl.load` and the single `tl.store` runs without touching HBM. The eager version's five kernels become one, five HBM round-trips become one, and five launches become one. The FLOP count is identical; the bytes and launches are what changed.

### A tiled GEMM: mapping output blocks onto SMs

Softmax is memory-bound, so the fused version wins on bytes. GEMM is different — it can be compute-bound at large batch — but the *structure* of a Triton matmul is the thing to understand, because it is how every high-performance kernel, including the quantized ones, is organized: tile the output, give each tile to one program, and stream the contraction dimension through SRAM.

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2-D grid: this program computes one BLOCK_M x BLOCK_N tile of C.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulate in fp32 for numerical stability, stream K in BLOCK_K chunks.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)          # compiles to tensor-core mma instructions
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c)


def triton_matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
    )
    return c
```

The launch grid `(cdiv(M, 128), cdiv(N, 128))` tiles the output matrix into 128×128 blocks and hands each block to a separate program instance, which the hardware schedules onto a streaming multiprocessor. The figure below shows exactly this mapping.

![Grid diagram of a 3 by 3 tiled GEMM where each output block C indexed by row and column is one program instance scheduled onto a separate streaming multiprocessor](/imgs/blogs/custom-cuda-kernels-for-inference-5.webp)

Each program loads a strip of $A$ and a strip of $B$ for the current $K$-chunk into SRAM, calls `tl.dot` (which the compiler lowers to tensor-core `mma` instructions), accumulates in FP32 registers, advances the pointers, and repeats until the contraction dimension is exhausted — then writes its tile of $C$ once. This is the skeleton. A production kernel adds `tl.constexpr` autotuning over block sizes, `num_warps`, `num_stages` for software pipelining, and group-ordering of program IDs for L2 cache reuse. But the load-tile / accumulate / store-tile shape never changes, and it is why quantized-GEMM kernels are "just" this loop with a dequantize step inserted right after the weight `tl.load`.

### Autotuning: the block sizes are not obvious

The `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages` in that kernel are not universal constants — the best values depend on the problem shape and the GPU. `num_warps` sets how many warps (groups of 32 threads) cooperate on one program's tile; more warps means more parallelism but also more register pressure. `num_stages` sets the depth of software pipelining: with `num_stages=4`, the compiler prefetches the next three $K$-chunks' loads while the tensor cores work on the current one, hiding global-memory latency behind compute. Get these wrong and you either spill registers to local memory (catastrophic) or leave the tensor cores stalling on loads. Triton's `@triton.autotune` sweeps a set of configs the first time it sees a new shape and caches the winner:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=5),
    ],
    key=["M", "N", "K"],   # re-autotune when these change; cache the winner per shape
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, *strides, **meta):
    ...
```

The `key=["M", "N", "K"]` is important for serving: it means the kernel re-tunes when the shape changes and reuses the cached config when it does not. This is exactly why hand-written kernels want *stable shapes* — if `M` (the batch-times-sequence dimension) churns every request, you either pay autotuning cost repeatedly or pad to a fixed bucket. Serving engines bucket sequence lengths and batch sizes precisely so the autotuner cache stays warm. If your shapes are genuinely dynamic and unpredictable, a hand-tuned Triton kernel loses much of its edge and you are better off with a library kernel that already covers the shape space or with `torch.compile`'s dynamic-shape handling.

### Triton vs CUDA C++ / CUTLASS: which and when

Triton is not always the answer. The trade is real:

| Approach | Dev effort | Control | Portability | Typical use |
|---|---|---|---|---|
| PyTorch eager | none | none | maximal | meeting SLO already; prototyping |
| `torch.compile` (Inductor→Triton) | low | medium (auto-fusion) | good | fusing pointwise chains; first thing to try |
| Hand-written Triton | medium | high (tiling, SRAM) | good (NVIDIA + AMD) | model-specific fused ops, attention variants |
| CUDA C++ / CUTLASS | high | maximal (warp, mma) | NVIDIA only, per-arch | quantized GEMM, peak-FLOP attention |
| PTX / SASS | extreme | absolute | one arch | a handful of library hot loops |

The decision rule I use: start with `torch.compile`; if the profile still shows an obvious fusion the compiler missed or an operator with no good library kernel, write it in Triton; drop to CUDA/CUTLASS only when you need tensor-core scheduling that Triton cannot express (fine-grained FP8 accumulation, exotic quantized layouts, warp-specialized producer/consumer pipelines like FlashAttention-3). In practice, the last category is written *once* by library authors — you consume Marlin and FlashAttention, you do not re-implement them.

## Fused decode kernels: the rest of the transformer

FlashAttention gets the headlines, but a transformer decode step spends real time outside attention, and every one of those operators is memory-bound and elementwise — the ideal fusion target. A serving engine ships a handful of small fused kernels that, together, remove most of the launch and HBM tax the earlier worked example counted. The important ones:

**Fused residual-add + RMSNorm.** Every transformer block does `h = x + residual` then `y = rmsnorm(h) * weight`, and then it *also* needs `h` itself as the next block's residual. Eager PyTorch runs this as an add (read two tensors, write one), a square-and-mean, an rsqrt, and two multiplies — five-plus kernels and several HBM passes over a tensor that fits nowhere useful in between. Fused, it is one kernel:

```python
@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr,
    row_stride, n_cols, eps, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # One HBM read of x and the incoming residual.
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)

    h = x + r                                            # fused residual add
    tl.store(res_out_ptr + row * row_stride + cols, h, mask=mask)  # new residual out

    var = tl.sum(h * h, axis=0) / n_cols                 # RMS in registers
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (h * rstd) * w                                   # normalize + scale

    # One HBM write of the normalized output (plus the residual write above).
    tl.store(out_ptr + row * row_stride + cols, y.to(tl.float16), mask=mask)
```

One row per program, everything between the loads and stores in registers. This is `torch.ops._C.fused_add_rms_norm` in vLLM, and it is a textbook memory-bound fusion: five-plus launches become one, and the tensor makes one trip across the bus instead of several.

**Fused RoPE.** Rotary position embeddings rotate pairs of dimensions in Q and K by position-dependent angles. Done eagerly, it is a sequence of slices, sin/cos multiplies, and a concatenate — several kernels over Q and K. Fused, one kernel reads Q and K, applies the rotation in registers using precomputed cos/sin tables, and writes them back once. It is frequently fused *further* into the attention kernel's Q/K load or into the QKV-projection epilogue, so the rotated tensors never touch HBM as a separate step at all.

**Fused SiLU-and-mul (SwiGLU gate).** The gated MLP computes `silu(gate_proj(x)) * up_proj(x)`. The activation-and-multiply half is pure elementwise: eager runs a SiLU (read/write) then a multiply (read two, write one). vLLM's `silu_and_mul` kernel reads the two halves of the gate/up output once, computes `silu(a) * b` in registers, and writes the result once — halving the HBM traffic of that step. The same pattern covers GeLU-and-mul for GeLU-gated models.

**Paged-attention kernel.** This one deserves emphasis because it is *why serving needs its own attention kernel distinct from FlashAttention*. In [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), the KV cache is not a contiguous tensor — it is split into fixed-size blocks (say 16 tokens) scattered across a memory pool, addressed through a per-sequence block table, exactly like OS virtual memory. A standard FlashAttention kernel assumes contiguous K and V; the paged-attention decode kernel instead *gathers* KV blocks by following the block table's indices, then runs the same online-softmax attention over the gathered blocks. That gather-through-an-indirection is the entire reason vLLM, SGLang, and TGI carry a bespoke paged-attention kernel rather than only calling the reference FlashAttention: the serving-time KV layout is paged and ragged, and no training-oriented kernel handles it. FlashInfer generalizes this further with a JIT kernel generator that specializes to the exact page size, shared-prefix structure, and batch composition at runtime.

The pattern across all four: they are memory-bound, elementwise-ish, and model-specific enough that a generic compiler sometimes leaves a fusion on the table — which is exactly the profile that justifies a hand-written kernel. None of them changes a FLOP; each removes launches and HBM round-trips.

For quick orientation, here is where each transformer operator's kernel comes from in a modern serving stack, and what replaces the naive eager version:

| Operator | Naive eager | Serving kernel | Where it comes from |
|---|---|---|---|
| Attention (prefill) | 3 kernels, $S{\times}S$ in HBM | FlashAttention-2/3 | library (`flash-attn`, SDPA) |
| Attention (paged decode) | dense copy + attention | paged-attention / FlashInfer | vLLM `_C`, FlashInfer |
| Residual add + RMSNorm | 5+ elementwise kernels | fused add-RMSNorm | Triton / vLLM `_C` |
| RoPE | slice + sin/cos + concat | fused rotary embedding | vLLM `_C`, often folded into QKV |
| SwiGLU gate | SiLU + multiply | silu-and-mul | vLLM `_C` |
| Projection GEMM (fp16) | `torch.matmul` | cuBLAS / CUTLASS | vendor library |
| Projection GEMM (INT4) | dequant + fp16 GEMM | Marlin W4A16 | library kernel |
| Projection GEMM (FP8) | fp16 GEMM | FP8 tensor-core GEMM | CUTLASS / TensorRT-LLM |

Read down the "serving kernel" column and the theme is unmistakable: almost every entry is a *fused* or *specialized* kernel that a library already ships, and the only column where you might author something is the paged/model-specific attention row. That is the map of where hand-written kernel effort actually lands.

## Shipping a custom kernel: the torch.library custom op

A fast kernel that PyTorch does not know about is a liability. If you call it as an opaque Python function, `torch.compile` will graph-break around it (killing the fusions and CUDA graphs on either side), and autograd and meta-tensor tracing will not know its shapes. The correct way to expose a hand-written kernel to the PyTorch ecosystem is `torch.library.custom_op`, which registers it as a first-class operator with a **fake (meta) implementation** so the compiler can trace shapes without running the kernel. The lifecycle is shown below.

![Timeline of shipping a custom kernel: write the kernel, JIT or nvcc compile, wrap as a torch.library op, register a fake meta implementation, wire into vLLM, then benchmark](/imgs/blogs/custom-cuda-kernels-for-inference-6.webp)

```python
import torch


@torch.library.custom_op("myops::fused_add_rmsnorm", mutates_args=())
def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    # Fuses residual add + RMSNorm into one kernel: one HBM read of x and
    # residual, one write of the normalized sum. _rmsnorm_launch dispatches to
    # the Triton or CUDA kernel authored above.
    return _rmsnorm_launch(x, residual, weight, eps)


@fused_add_rmsnorm.register_fake
def _(x, residual, weight, eps):
    # Shape/dtype-only "fake" run so torch.compile can trace without the GPU.
    return torch.empty_like(x)


# Now it composes cleanly: the compiler can fuse around it and capture it in a
# CUDA graph, because it knows the op's output shape from the fake impl.
compiled = torch.compile(model, mode="reduce-overhead")
```

The `register_fake` decorator is the load-bearing part. Without it, `torch.compile` cannot know the output shape, so it must fall back to eager for the surrounding region — you lose the fusion and CUDA-graph capture that would otherwise have amortized the launch cost. `mutates_args=()` tells the compiler the op has no in-place side effects, which unlocks more aggressive reordering. (Fused residual-add-RMSNorm is deliberately declared *not* mutating here; if you write the normalized result back into `x` in place, you must declare `mutates_args=("x",)` or you will get silently wrong results after compilation.)

### How vLLM, SGLang, and TGI wire kernels in

Production engines do not ask you to register ops one at a time; they ship a compiled extension of custom kernels and dispatch to them internally. It is worth knowing where they live:

- **vLLM** compiles a C++/CUDA extension (`csrc/`) exposing ops under `torch.ops._C` — paged-attention, fused RMSNorm, RoPE, SiLU-and-mul, and the quantized GEMMs (GPTQ/AWQ/Marlin/FP8). The engine's model definitions call these directly instead of the eager `aten` ops. FlashAttention and FlashInfer are optional backends selected at runtime.
- **SGLang** uses FlashInfer as its primary attention backend and ships its own `sgl-kernel` package of fused kernels; its RadixAttention prefix cache is built on paged KV kernels.
- **TGI** (Text Generation Inference) ships flash-attention, paged-attention, and quantized (GPTQ/AWQ/EETQ) kernels as part of its server image, dispatched by dtype and hardware.

The high-level API hides all of this. Serving a GPTQ checkpoint with the Marlin kernel is a one-line flag:

```python
from vllm import LLM, SamplingParams

# vLLM detects the GPTQ 4-bit checkpoint and, on Ampere+ GPUs, routes every
# projection matmul through the Marlin W4A16 kernel automatically.
llm = LLM(
    model="TheBloke/Llama-2-13B-chat-GPTQ",
    quantization="gptq_marlin",   # explicit; "gptq" auto-upgrades to Marlin when supported
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
out = llm.generate(["Explain why decode is memory-bound."], params)
print(out[0].outputs[0].text)
```

Under this call, the Marlin kernel is invoked as `torch.ops._C.gptq_marlin_gemm(...)` for each linear layer — the same custom-op mechanism as above, just registered by the library rather than by you. You get the speedup without writing a line of CUDA, which is exactly the point of the abstraction ladder: drop only as far as you must.

## Quantized GEMM kernels: Marlin, W4A16, and FP8

Quantization is the highest-leverage kernel work for decode, because decode is memory-bound and quantization directly cuts the bytes. But there is a subtlety that trips up naive implementations, and it is worth seeing clearly. The figure below contrasts the naive approach with Marlin.

![Before-after comparison of a naive dequantize-then-fp16-GEMM that writes fp16 weights to HBM versus Marlin fused W4A16 that dequantizes INT4 weights in registers](/imgs/blogs/custom-cuda-kernels-for-inference-7.webp)

The naive way to serve a 4-bit model is: dequantize the INT4 weights to FP16 (a kernel that reads 4-bit, writes 16-bit to HBM), then run a normal FP16 GEMM (which reads those 16-bit weights back). You have now moved the weights across HBM as FP16 — you got the storage savings but *threw away the bandwidth savings*, which is the only thing that helps decode. Worse, at batch > 1 the naive INT4 kernels in early implementations collapsed to FP16 speed because they were designed only for the batch-1 GEMV case.

**Marlin** (Frantar et al., 2024, from IST-DASLab) solves this by keeping the weights 4-bit in HBM and dequantizing *in registers* inside the GEMM loop, right after the `tl.load`/`ld.global` of the packed 4-bit tile — the dequantize step sits between loading the quantized tile and the `mma`. The weights are read once, as 4-bit, so decode gets the full ~4× bandwidth reduction. Marlin's key engineering is sustaining that speedup across batch sizes up to 16–32 through careful pipelining and layout, where naive kernels degrade. The reported result is near-ideal: roughly **3.87× over FP16** at the memory-bound end, holding up through moderate batch.

#### Worked example: what "dequantize in registers" actually saves

Take a single 4096×4096 weight matrix. In FP16 it is $4096^2 \times 2 = 33.5$ MB. In INT4 with group-wise scales (one FP16 scale per group of 128 weights along the contraction dimension) it is $4096^2 \times 0.5 = 16.8$ MB of packed 4-bit weights plus $4096 \times (4096/128) \times 2 = 0.26$ MB of scales — about 17 MB total, roughly half the FP16 bytes. The naive path loads that 17 MB, writes 33.5 MB of dequantized FP16 to HBM, then the GEMM reads the 33.5 MB back: **84 MB of traffic**. Marlin loads the 17 MB of INT4 weights and scales once, unpacks and scales them in registers just before the `mma`, and never writes FP16 weights out: **17 MB of traffic**. For memory-bound decode, time tracks bytes, so ~5× less traffic on the weight term is the ~4× end-to-end you observe. The dequantize itself — a shift, a mask, a subtract of the zero-point, and a multiply by the group scale — is a handful of FLOPs per weight, invisible against the bandwidth it saves.

The group-scale layout is why these kernels are fiddly: the packed 4-bit weights are pre-shuffled into a hardware-friendly order at load time (Marlin repacks GPTQ/AWQ checkpoints into its own layout) so that a warp's `mma` operands land in the right lanes without a transpose, and the scales are interleaved so each dequantized tile finds its scale in the same cache line. This is the kind of detail you do not want to reimplement, and the reason "consume the library" is the right default for quantized GEMM.

The landscape of quantized GEMM kernels:

| Kernel | Scheme | Weights / activations | Hardware | Reported speedup vs FP16 |
|---|---|---|---|---|
| cuBLAS FP16 | none | FP16 / FP16 | any CUDA | 1.0× (baseline) |
| Marlin | GPTQ/AWQ W4A16 | INT4 / FP16 | Ampere, Hopper | ~3.5–3.9× (decode) |
| Machete | W4A16 (CUTLASS) | INT4 / FP16 | Hopper | ~3–4×, better at larger batch |
| Marlin-FP8 / W8A8 | FP8 | FP8 / FP8 | Hopper (H100) | ~1.8–2× (compute-bound too) |
| TensorRT-LLM FP8 | FP8 E4M3 | FP8 / FP8 | Hopper | ~2× + tensor-core throughput |

The distinction between W4A16 and FP8 matters and maps to the roofline. **W4A16** (4-bit weights, 16-bit activations) attacks the *memory-bound* decode regime — it cuts weight bytes, which is what decode is starved on. **FP8** (8-bit weights *and* activations) additionally attacks the *compute-bound* prefill regime, because H100 tensor cores run FP8 GEMM at roughly double the FP16 rate. If you serve long prompts or large batches, FP8 buys you compute throughput that W4A16 alone does not. If you serve short-prompt, batch-1-ish chat, W4A16 is the bigger win because you are almost purely bandwidth-bound. The accuracy trade-offs and calibration for these schemes are covered in [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving); the kernel point here is that the *scheme* and the *kernel* are coupled — you pick the kernel that turns your quantization format into fewer bytes on your hardware.

## Measuring the win: kernel-level and end-to-end

Never ship a kernel you have not benchmarked, and never trust a kernel-level microbenchmark as an end-to-end number. Two measurements matter: the isolated kernel time (does this kernel actually beat what it replaces, at the shapes I run?), and the end-to-end tokens/s (did the win survive the rest of the pipeline?).

For the kernel level, `triton.testing.do_bench` handles warmup, CUDA-graph-free timing, and clock-quantile reduction correctly — do not roll your own with `time.time()` around a single call, because you will measure launch queueing, not kernel time.

```python
import torch
import triton


def bench(fn, *args, **kwargs):
    # do_bench warms up, flushes the L2 between reps, and returns median ms.
    return triton.testing.do_bench(lambda: fn(*args, **kwargs), warmup=25, rep=100)


x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

t_eager = bench(lambda z: torch.softmax(z, dim=-1), x)
t_triton = bench(triton_softmax, x)

# Effective bandwidth: softmax reads x once and writes y once => 2 * bytes.
bytes_moved = 2 * x.numel() * x.element_size()
bw_eager = bytes_moved / (t_eager * 1e-3) / 1e9
bw_triton = bytes_moved / (t_triton * 1e-3) / 1e9

print(f"eager   {t_eager:.3f} ms   {bw_eager:6.0f} GB/s")
print(f"triton  {t_triton:.3f} ms   {bw_triton:6.0f} GB/s")
print(f"speedup {t_eager / t_triton:.2f}x")
```

The bandwidth line is the honest metric for a memory-bound kernel: divide bytes moved by time and compare to the card's peak (2.0 TB/s on A100, 3.35 TB/s on H100). A fused softmax should land close to peak; if it does not, you are leaving bytes on the table (bad tiling, uncoalesced access, register spills). A kernel already at 90% of peak bandwidth cannot be meaningfully sped up — the roofline says so — and that is your signal to stop optimizing and move on.

End-to-end, benchmark the *serving metrics*: TTFT, TPOT, and aggregate tokens/s under realistic concurrency, using the engine's own benchmark harness (`vllm bench` / `benchmark_serving.py`). The figure below shows the cumulative effect of stacking these kernels on a Llama-3-8B service, and the markdown table restates it as the named-hardware before→after that any measurement block in this series must carry.

![Matrix showing cumulative TTFT, TPOT, decode tokens per second, and GPU utilization improvements from PyTorch eager through FlashAttention-2, fused norms, Marlin W4A16, and FP8 kernels on Llama-3-8B](/imgs/blogs/custom-cuda-kernels-for-inference-8.webp)

| Configuration (Llama-3-8B) | GPU | TTFT | TPOT | Decode tok/s | GPU util |
|---|---|---|---|---|---|
| PyTorch eager baseline | A100 80GB | ~210 ms | ~38 ms | ~1,150 | ~55% |
| + FlashAttention-2 | A100 80GB | ~95 ms | ~26 ms | ~1,900 | ~68% |
| + fused RMSNorm + RoPE + SwiGLU | A100 80GB | ~88 ms | ~22 ms | ~2,250 | ~74% |
| + Marlin W4A16 | A100 80GB | ~70 ms | ~14 ms | ~3,400 | ~82% |
| + FP8 kernels | H100 80GB | ~40 ms | ~9 ms | ~6,200 | ~88% |

These figures are illustrative and order-of-magnitude — the exact numbers depend on prompt length, batch size, and framework version — but the *shape* is what production consistently shows: fused attention delivers the first large step (it removes the $O(N^2)$ HBM tax), fused elementwise kernels add a steady margin (they remove launches and round-trips), and quantized GEMM plus the H100 move compound on top. The aggregate is roughly a 5× improvement in decode throughput on the same class of GPU.

#### Worked example: what the kernels do to cost per million tokens

Cost per token is (GPU \$/hour) divided by (tokens/hour). At an illustrative A100 80GB on-demand rate of \$1.80/hour, the eager baseline at ~1,150 tok/s produces $1150 \times 3600 = 4.14$M tokens/hour, so \$0.43 per 1M tokens. With Marlin at ~3,400 tok/s that is 12.2M tokens/hour and \$0.15 per 1M — the kernel work alone made serving roughly **3× cheaper on the identical card**. Moving to H100 at an illustrative \$3.20/hour and ~6,200 tok/s gives 22.3M tokens/hour and \$0.14 per 1M: a more expensive GPU at essentially the same unit cost, because the throughput scaled with the price. That is the whole economic argument for kernel work at scale — it does not just cut latency, it moves the \$/1M-token number that finance actually tracks. At a billion tokens a day, the gap between \$0.43 and \$0.14 is roughly \$100k a month, which is what justifies a kernel specialist.

## Profiling: find the kernel that actually matters

Everything above assumes you already know which kernel to optimize. In practice, the first and most valuable skill is *measuring where the time goes*, because intuition about GPU performance is reliably wrong. Two NVIDIA tools do the job, at two granularities.

**Nsight Systems (`nsys`)** captures the whole timeline: kernel launches, gaps between them, memory copies, and CPU-side Python overhead. This is where you diagnose *launch-bound* execution — the 2 a.m. symptom. If the timeline shows thin kernels separated by wide gaps, you are launch-bound and the fix is fusion or CUDA graphs, not a faster kernel. If the kernels are packed edge to edge and GPU utilization is genuinely high, you are compute- or bandwidth-bound and need a better kernel. A typical capture is one line:

```bash
nsys profile -o decode_trace --trace=cuda,nvtx \
  python bench_decode.py --model llama-3-8b --batch 1 --tokens 128
```

**Nsight Compute (`ncu`)** profiles a single kernel in depth: achieved vs peak bandwidth, tensor-core utilization, occupancy, register spills, and — most useful — an automatic roofline placement telling you whether *this specific kernel* is memory- or compute-bound and how close it is to the relevant ceiling. The stopping criterion falls right out of it: a memory-bound kernel already at 90%+ of peak HBM bandwidth cannot be meaningfully improved, and a compute-bound kernel near peak tensor-core throughput is done. Chasing a kernel that ncu reports at 92% of peak bandwidth is wasted effort; the roofline says there is nothing left.

```bash
ncu --set full --kernel-name regex:"fused_add_rmsnorm|softmax" \
  --launch-count 3 python bench_decode.py
```

For a first pass without leaving Python, `torch.profiler` with `ProfilerActivity.CUDA` and `sort_by="cuda_time_total"` gives a ranked table of where GPU time went — enough to find the top three kernels before you reach for the NVIDIA tools. Whichever tool you use, the governing law is Amdahl's: a 10× speedup on an operator that is 3% of the step is a 2.7% end-to-end win, and not worth a week. Optimize the top of the profile, re-measure, and stop when the next candidate is single-digit percent. Profiling is also how you catch regressions like ours — a golden `nsys` trace diffed against production is a cheap guard against a "safe" change silently disabling a fused path.

## Portability: one kernel does not fit every GPU

The hidden cost of a hand-written kernel is that it is tuned for one architecture, and the fleet is rarely one architecture. The three you are most likely to serve on differ in ways that break assumptions baked into a kernel:

- **A100 (Ampere).** 108 SMs, up to 192 KB shared memory per SM, FP16/BF16 tensor cores, 2.0 TB/s HBM2e. No FP8, no TMA. FlashAttention-2 is the peak attention kernel here; FP8 kernels and FlashAttention-3 simply do not run.
- **H100 (Hopper).** 132 SMs, up to 228 KB shared memory, FP8 tensor cores, the Tensor Memory Accelerator (TMA) for asynchronous bulk copies, `wgmma` warp-group matmul, and thread-block clusters. FlashAttention-3 and FP8 GEMM exploit all of these; a kernel that hard-codes Ampere assumptions leaves most of the H100 on the table.
- **MI300X (AMD CDNA3).** 192 GB HBM3, matrix cores instead of tensor cores, programmed through ROCm/HIP rather than CUDA. A CUDA C++ kernel does not run here at all; it needs a HIP port. This is Triton's structural advantage — it retargets to AMD's backend from the same `@triton.jit` source, so a Triton kernel is far more portable than a CUDA one.

The consequences for the decision: a CUDA/CUTLASS kernel is a *per-architecture maintenance commitment* — someone must re-tune (and possibly re-port) it every time the fleet gains a new GPU, and a kernel tuned for H100's larger shared memory and `wgmma` can be *slower* than the library fallback on A100. Triton softens this by retargeting, but block sizes and `num_stages` still want re-autotuning per architecture. Library kernels (FlashAttention, FlashInfer, Marlin, cuBLAS) handle this dispatch for you — they ship an entry per architecture and pick at runtime, which is the deeper reason "consume libraries" is the default. Write your own only for an operator no library covers, and prefer Triton over CUDA unless you specifically need Hopper-only instructions the library authors are already using.

## Case studies

**FlashAttention → FlashAttention-2 → FlashAttention-3.** The original FlashAttention (Dao et al., 2022) reported up to 3× speedup on GPT-2 and enabled context lengths that previously OOM'd, purely by not materializing the score matrix. FlashAttention-2 (Dao, 2023) restructured the work partitioning across warps and reduced non-matmul FLOPs, reaching roughly **2× over FlashAttention-1** and sustaining 50–73% of A100 theoretical peak (up to ~230 TFLOPS). FlashAttention-3 (Shah et al., 2024) targets Hopper specifically: it uses warp-specialized producer/consumer pipelines, the Tensor Memory Accelerator (TMA), and FP8, reaching about **75% of H100's FP16 peak (~740 TFLOPS)** and roughly 1.2 PFLOPS with FP8 — around 1.5–2.0× over FlashAttention-2 on H100. The through-line: each generation is the *same attention math*, re-scheduled to move fewer bytes and feed the tensor cores more continuously on newer silicon.

**FlashInfer.** FlashInfer (Ye et al., MLSys 2025, which received a best-paper award) is a library of composable, block-sparse attention kernels built specifically for *serving*, not training. It handles paged KV cache layouts, shared prefixes, and heterogeneous batch shapes with a JIT-compiled kernel generator, and it is now a backend in vLLM, SGLang, and TGI. The reported serving wins are in inter-token latency reduction (tens of percent) versus prior attention backends, precisely because it specializes the kernel to the *serving-time* KV layout — the paged, ragged, prefix-shared cache that training-oriented kernels never see. This is a clean example of why serving needs its own kernels: the shapes are different from training.

**SGLang and RadixAttention.** SGLang's contribution is a kernel-level realization of prefix sharing. RadixAttention stores the KV cache in a radix tree keyed on token prefixes, so many requests that share a system prompt or a few-shot preamble share the *same* physical KV blocks — and the attention kernel reads those shared blocks once rather than per request. The kernel work is the paged-attention gather again, but over a tree-structured cache with reference counting for eviction. For agentic and chat workloads where thousands of requests share a long system prompt, this turns a large fraction of the KV read (the dominant decode cost we computed earlier) into a cache hit. It is another instance of the same principle: the win came from a kernel that matches the serving-time data structure, not from faster arithmetic. The prefix-sharing mechanics are covered in the [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) post; here the point is that RadixAttention is fundamentally a *kernel plus a data structure* co-designed for serving.

**The production reality.** None of these is exotic anymore. A default vLLM install on an H100 today already dispatches to FlashAttention-3 or FlashInfer for attention, ships `fused_add_rms_norm`, `rotary_embedding`, and `silu_and_mul` as compiled ops in `torch.ops._C`, and routes quantized checkpoints through Marlin or FP8 GEMM — all without you writing a line of CUDA. The 5× in the measurement table is not a research result you have to reproduce; it is roughly what you get by *turning the features on* and letting the engine pick kernels. The residual opportunity for hand-written kernels is the model-specific operator (a novel attention variant, a fused MoE router, a custom positional scheme) that the library menu does not yet cover — and even then, the right first move is often to contribute it upstream rather than carry it privately.

**Marlin.** Marlin's contribution is not a new quantization *scheme* but a kernel that finally *realizes* the theoretical W4A16 bandwidth savings in practice, and sustains them past batch 1. Prior 4-bit kernels hit ~4× at batch 1 and then collapsed toward FP16 as batch grew (because they dequantized to FP16 and became compute-bound on the FP16 path). Marlin keeps the weights 4-bit through the load and dequantizes in registers, holding roughly **3.87× near-ideal speedup up to batch 16–32**. It shipped in vLLM as the default kernel for GPTQ/AWQ checkpoints on Ampere and Hopper, and is the reason "4-bit for throughput" is a real strategy rather than a batch-1 curiosity.

## Common failure modes when writing kernels

Having shipped and debugged a few of these, the failures cluster into a short list. Every one has cost me a night, and none is exotic.

**Numerical mismatch from accumulation dtype.** The most common correctness bug is accumulating a reduction (a softmax sum, a GEMM's $K$-loop, a norm's variance) in FP16 instead of FP32. FP16 has ~10 bits of mantissa; summing thousands of terms in it drifts, and your kernel produces subtly wrong logits that a unit test on a tiny input will not catch but a long-context generation will. Always accumulate in FP32 (`tl.zeros(..., dtype=tl.float32)`), cast to the storage dtype only at the final store. Validate against the eager reference with a tolerance, not exact equality — a good target is max absolute error under ~1e-2 for FP16 and matching argmax on the logits.

**Silent corruption from a wrong `mutates_args`.** If your custom op writes into one of its inputs in place but you declare `mutates_args=()`, `torch.compile` is free to reorder or elide the write, and you get results that are correct in eager and wrong after compilation — the worst kind of bug, because it only appears in the fast path you ship. Declare every mutated tensor. When in doubt, write to a fresh output and declare no mutation; the extra allocation is cheaper than the incident.

**Graph breaks from a missing fake impl.** Forget `register_fake` and `torch.compile` cannot trace your op's shapes, so it falls back to eager around it, killing the fusions and CUDA-graph capture on both sides. The kernel is fast in isolation and the end-to-end number barely moves. Always register the fake impl, and verify with `torch.compile(fullgraph=True)` in a test — it will raise on any graph break rather than silently degrading.

**Autotune cache thrash on dynamic shapes.** An autotuned kernel re-tunes whenever its `key` dimensions change. If batch-times-sequence churns every request, you pay tuning cost repeatedly and your p99 spikes on cold shapes. Bucket shapes (pad to the next power of two, or a fixed set of batch sizes) so the cache stays warm, or accept a single non-autotuned config chosen for your dominant shape.

**A correct kernel that is slower than the library.** It is entirely possible to write a kernel that passes correctness and is *slower* than the cuBLAS or FlashAttention path it replaced, especially at shapes you did not tune for or on a GPU generation you did not test. This is why the benchmark is not optional and why the engine must keep a library fallback: your kernel should be dispatched only for the shape/arch cells where you have measured a win.

**Register spills and low occupancy.** Ask for too large a tile or too many warps and the kernel spills registers to local memory (which lives in HBM), quietly turning a compute-bound kernel back into a memory-bound one. `ncu` reports register count and occupancy; if occupancy is low and there are spills, shrink the tile. The symptom is a kernel that looks reasonable but sits far below its roofline ceiling.

| Symptom | Likely cause | Fix |
|---|---|---|
| Wrong long-context output, right short output | FP16 accumulation drift | accumulate in FP32 |
| Correct eager, wrong after `torch.compile` | wrong `mutates_args` | declare mutated tensors |
| Fast kernel, flat end-to-end | missing `register_fake` (graph break) | register fake impl; test `fullgraph=True` |
| p99 spikes on new shapes | autotune cache thrash | bucket/pad shapes |
| Below roofline ceiling in `ncu` | register spill / low occupancy | shrink tile, fewer warps |

## When to use this (and when not to)

Custom kernels are not free, and most teams reach for them too early. Here is the decision, stated plainly.

**Reach for library kernels (FlashAttention, FlashInfer, Marlin, FP8) essentially always.** These are not "custom kernels" in the sense of work you do — they are a flag or an install. If you are serving LLMs and not using a fused attention backend and a quantized-GEMM kernel where the checkpoint allows, you are leaving 2–5× on the floor for zero engineering cost. This is the easy yes.

**Try `torch.compile` before writing any Triton.** For fusing your own pointwise chains and small custom operators, `torch.compile(mode="reduce-overhead")` with the Inductor backend generates Triton and captures CUDA graphs automatically. Write Triton by hand only when the profile shows the compiler missing a fusion you know is possible, or an operator with no good library kernel. Details of that compiler path are in the [torch.compile and CUDA graphs post](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile).

**Write a hand kernel (Triton, or CUDA via a library) when:** you have a model-specific fused operator on the hot path that no library covers (a novel attention variant, a fused MoE routing+gate, a custom RoPE); the operator is memory-bound and you have profiled that it is a real fraction of TPOT; and you have the shapes pinned down enough to specialize. The bar is a measured, material fraction of end-to-end latency — not "this looks slow."

**Do not write a custom kernel when:** you are meeting your SLO (the fastest kernel is the one you did not have to write or maintain); the operator is a tiny fraction of runtime (Amdahl's law — a 10× speedup on 3% of the time is a 2.7% end-to-end win); your shapes change constantly (autotuned kernels assume stable shapes); or you cannot commit to maintaining it across GPU generations. Portability is the hidden cost: a kernel hand-tuned for H100's tensor-core layout may be *slower* than the library fallback on A100 or MI300X, and someone has to keep it working when the next architecture ships. Triton mitigates this (it retargets to AMD), but CUDA/CUTLASS kernels are a per-architecture maintenance commitment. If your team cannot own that, consume libraries and stop.

The honest summary: the 80/20 is (1) turn on FlashAttention/FlashInfer, (2) quantize with a real kernel (Marlin or FP8), (3) `torch.compile`. Hand-written kernels are the last 20% for teams whose scale makes a few percent of GPU cost worth a specialist's quarter.

### The three questions before you write a kernel

When someone proposes a custom kernel, I ask three things, in order, and a "no" to any of them ends the conversation:

1. **Is it on the profile's top three?** Pull an `nsys`/`ncu` trace. If the operator is not a material fraction of end-to-end time, Amdahl's law caps the win below what the effort costs. A kernel that is 2% of the step is not worth writing no matter how much you can speed it up. If you cannot produce the trace, you are guessing, and the answer is no.
2. **Is it memory-bound with an avoidable HBM round-trip, or a shape no library covers?** These are the two profiles where hand-written kernels reliably win: fusing an elementwise chain that currently round-trips HBM, or serving a shape/layout (a paged variant, a novel attention mask) that no library kernel handles. If it is a plain GEMM at a common shape, cuBLAS or CUTLASS already beat what you will write — use them.
3. **Can we own it across GPU generations and shape churn?** A kernel is a liability with a maintenance tail: it must be re-tuned per architecture, kept correct through framework upgrades, and given a library fallback for every cell it does not cover. If the team cannot commit to that — or the shapes are too dynamic to autotune — the kernel will rot into a slow path nobody dares touch. Prefer Triton for portability, and prefer contributing upstream over carrying a private fork.

Pass all three and you have a real candidate: profile it, wrap it as a `torch.library` op with a fake impl, benchmark kernel-level and end-to-end, dispatch it only where you measured the win, and keep the library fallback. Fail any one and the right move is to consume a library and spend the quarter somewhere with more leverage.

## Key takeaways

- **Decode is memory-bound, so bytes moved is the metric that matters.** Arithmetic intensity below the roofline ridge (~156 FLOP/byte on A100, ~295 on H100) means the tensor cores wait on HBM. Compute tuning does nothing; cutting HBM traffic does everything.
- **Fusion wins by not writing intermediates to HBM.** FlashAttention is faster with identical FLOPs because it never materializes the $N \times N$ score matrix — $\Theta(N^2 d^2 M^{-1})$ HBM accesses instead of $\Theta(Nd + N^2)$.
- **Kernel-launch overhead is a second, independent tax.** ~5–10 µs × hundreds of kernels per token can be 1–3 ms of pure overhead; fusion removes launches, CUDA graphs make the rest cheap.
- **Climb only as far down the ladder as you must.** PyTorch eager → `torch.compile`/Triton → CUDA/CUTLASS → PTX. Most inference wins are the middle two rungs; PTX is almost never worth it.
- **A custom kernel must be a `torch.library` op with a `register_fake` meta impl,** or `torch.compile` graph-breaks around it and you lose the fusions and CUDA graphs you were trying to keep.
- **W4A16 attacks memory-bound decode; FP8 additionally attacks compute-bound prefill.** Pick the quantized-GEMM kernel that turns your format into fewer bytes on your hardware — Marlin for 4-bit decode, FP8 for H100 prefill/large batch.
- **Benchmark kernel-level *and* end-to-end.** Use `triton.testing.do_bench` for the kernel and the engine's serving harness for TTFT/TPOT/tok/s. A kernel at 90% of peak bandwidth cannot be sped up — stop.
- **Consume libraries; hand-write rarely.** FlashAttention, FlashInfer, and Marlin are a flag away and give 2–5× for free. Hand-written kernels are for a profiled hot-path operator no library covers, at a scale where a few percent of GPU cost pays for a specialist.

## Further reading

- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022) — the I/O-complexity argument and the online-softmax tiling.
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023) — the 2× rework and A100 peak-FLOP results.
- Shah, Bikshandi, Zhang, Thakkar, et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024) — Hopper warp-specialization, TMA, FP8.
- Ye, Chen, Lai, et al. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving* (MLSys 2025) — composable, paged, serving-specific attention kernels.
- Frantar, Castro, Chen, Hoefler, Alistarh. *Marlin: a Mixed-precision Auto-regressive Parallel Inference kernel* / GPTQ line of work (IST-DASLab, 2024) — sustained W4A16 speedup past batch 1.
- Tillet, Kung, Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (2019) — the DSL used above; see the official Triton tutorials for fused softmax and matmul.
- NVIDIA CUTLASS documentation and the [CUDA programming for AI engineers](/blog/machine-learning/high-performance-computing/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel) primer — threads, blocks, tiling, and a first kernel from scratch.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) (the SLO triangle), [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) (the memory-wall spine), and the sibling [kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) (the compiler-generated path).
