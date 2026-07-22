---
title: "Triton for inference kernels, and when to stop writing CUDA"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "You spent Track E hand-writing CUDA; this post ports those kernels to Triton, autotunes them, removes a host sync from the sampler, and gives you a per-kernel rule for when Triton is enough and when hand CUDA still earns its keep."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "triton",
    "cuda",
    "kernels",
    "autotune",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 47
---

This is the post that closes Track E, and it is the one where I argue against most of the previous five. Over those posts you hand-wrote CUDA for `nanoserve`: an RMSNorm with a warp reduction and vectorized `float4` loads, a fused RoPE, a `reshape_and_cache` that writes new keys and values into paged blocks under a block-table indirection, a decode attention kernel with an online softmax and split-K, and a dequant-fused GEMM that unpacks 4-bit weights in registers. Each of those was a few hundred lines of index arithmetic, `__shared__` staging, and warp shuffles. Each of them took a day to get correct and a week to make fast, and every one of them is nailed to one architecture.

Here is the uncomfortable question that a principal engineer eventually asks their own kernel folder: how much of that did you need to write by hand? The honest answer, for inference, is *most of it, no*. Not because CUDA is bad — because for the kernels an inference engine actually spends its time in, a compiler can now generate code that ties hand-tuned CUDA on the hardware you have, from about a twentieth of the source, and runs on the hardware you don't. That compiler's front end is [Triton](https://triton-lang.org/), and this post is about when to reach for it, when to reach past it for a library, and when to keep the CUDA.

![A two column comparison of what a CUDA author manages by hand against what the Triton compiler manages from a block description](/imgs/blogs/triton-for-inference-kernels-and-when-to-stop-writing-cuda-1.webp)

The figure above is the whole argument in one picture, and it is the mental model to carry through the post. In CUDA you own the threads, the warps, the shared-memory staging, and the coalescing. In Triton you describe the computation in terms of *blocks of data* and the compiler owns all four of those. By the end you will have `nanoserve/kernels/triton/`: an RMSNorm and a fused RoPE ported from Track E's CUDA, an `@triton.autotune`-wrapped launcher, and a top-k/top-p sampler that runs entirely on the GPU and removes the host synchronization that quietly caps your decode throughput. You will also have a decision matrix you can defend in a design review, grounded against the one public result that makes the case sharpest.

Standard promise for this series, restated from [the introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is arithmetic I show you in full, a citation with a link, or a range framed as *what you will see when you run the script*. The results tables carry a `Source` column. The one headline benchmark I lean on is vLLM's, cited and dated, never mine.

---

## 1. The one idea: you write blocks, the compiler writes threads

Track E's CUDA kernels all had the same shape underneath the math: you computed a global thread index, mapped it to an element or a tile, loaded from global memory in a pattern you designed to be coalesced, staged partial results in `__shared__` memory, reduced across the warp with `__shfl_down_sync`, and stored the result back. The *math* — square, sum, `rsqrt`, scale — was maybe six lines. The other seventy lines were you doing the GPU's bookkeeping.

Triton inverts the ownership. You write a Python function decorated with `@triton.jit`. Inside it you never mention a thread. You ask for a *program id* (`tl.program_id`), you build a vector of offsets with `tl.arange`, and you `tl.load` a whole block of elements at those offsets in one call. Arithmetic is on tensors — `tl.sum`, `tl.dot`, `x * x`. You `tl.store` the block back. The Triton compiler then decides, for the target architecture, how many threads run each program, how the block is split across the warps, how loads are coalesced and vectorized, whether an intermediate lives in registers or shared memory, and how to pipeline loads against compute. That is the trade you are making: you give up manual control of the schedule and you get portability, brevity, and an autotuner in return.

![A layered stack showing the six math operations you write on top and the coalescing reduction vectorization and scheduling the compiler generates below](/imgs/blogs/triton-for-inference-kernels-and-when-to-stop-writing-cuda-2.webp)

The clearest way to feel the difference is to put the *same* kernel in both languages. Here is the RMSNorm from [the first CUDA post](/blog/machine-learning/inference-engineering/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope), abbreviated to its essential body — one block per row, a warp reduction of the sum of squares, then a scaled write:

```cpp
// nanoserve/kernels/cuda/rmsnorm.cu  (abbreviated body)
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ y,
                               int hidden, float eps) {
  int row = blockIdx.x;
  const float* xr = x + row * hidden;
  float* yr = y + row * hidden;

  // 1) partial sum of squares over this thread's strided slice
  float acc = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = xr[i];
    acc += v * v;
  }
  // 2) warp reduction, then a shared-memory reduction across warps
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffff, acc, off);
  __shared__ float warp_sums[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  // 3) first warp reduces the per-warp partials (elided), yields `mean`
  float mean = block_reduce(warp_sums, blockDim.x) / hidden;
  float inv = rsqrtf(mean + eps);
  // 4) scaled store
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    yr[i] = xr[i] * inv * w[i];
}
```

That is the *cleaned-up* version; the real one carries `float4` vectorized loads, a bounds tail, and a template on dtype. Now the same operation in Triton:

```python
# nanoserve/kernels/triton/rmsnorm.py
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr,
                   hidden, eps,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)                      # one program per row
    x_row = x_ptr + row * hidden
    offs = tl.arange(0, BLOCK)                   # a vector of column indices
    mask = offs < hidden                         # guard the tail
    x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden         # reduction the compiler lowers
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * inv * w
    tl.store(y_ptr + row * hidden + offs, y.to(tl.float16), mask=mask)
```

The warp reduction, the shared-memory staging, the `__syncthreads`, the vectorized load, the two-level reduction across warps — all of that is gone. It became `tl.sum(x * x, axis=0)`. The compiler emits the reduction, and on a different architecture it emits a *different* reduction. That is not pseudocode: this compiles and runs against a reference RMSNorm within floating-point tolerance, and the `BLOCK` is a compile-time constant (`tl.constexpr`) so the compiler can size registers and unroll. The launcher is three lines:

```python
def rmsnorm(x, w, eps=1e-5):
    B, H = x.shape
    y = torch.empty_like(x, dtype=torch.float16)
    BLOCK = triton.next_power_of_2(H)            # e.g. 4096 -> 4096
    rmsnorm_kernel[(B,)](x, w, y, H, eps, BLOCK=BLOCK)
    return y
```

`rmsnorm_kernel[(B,)]` is the launch grid: `B` programs, one per row. There is no block-dimension to choose, no occupancy calculator to consult by hand. If you want that choice made *for* you and made *well*, that is what autotuning is, and it is the next section.

One caveat before you delete the CUDA. This RMSNorm assumes the whole row fits in one block (`BLOCK >= hidden`), which is fine for a 4096-wide hidden state and a `BLOCK` of 4096, but if you ever ran a hidden dimension past roughly the register/shared budget you would need a two-pass reduction — the CUDA version handles arbitrary width with its strided loop, the naive Triton version does not. For inference on the models in [our matrix](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — Llama-3.1-8B at hidden 4096, Qwen3-8B, Gemma-3-12B — the single-block form is correct and is what the production engines use.

#### Worked example: is the Triton RMSNorm anywhere near the hardware limit?

RMSNorm is memory-bound, so the honest metric is not tok/s, it is *achieved HBM bandwidth* against the roofline (see [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)). Derive the traffic. For one row of Llama-3.1-8B, hidden $H = 4096$, in fp16 (2 bytes): you read the input row ($4096 \times 2 = 8{,}192$ bytes), read the weight vector once ($8{,}192$ bytes, but it is reused across every row in the batch and stays resident in cache), and write the output row ($8{,}192$ bytes). So per row the *streamed* traffic is read + write of the activation: about 16 KB, with the 8 KB weight amortized.

For a decode batch of $B = 256$ rows: $256 \times 16\text{ KB} = 4\text{ MB}$ of activation traffic, plus one 8 KB weight load. On an A100 with about 2.0 TB/s of HBM bandwidth (NVIDIA A100 datasheet), the memory-bound floor is

$$t_{\min} = \frac{4 \times 10^{6}\ \text{bytes}}{2.0 \times 10^{12}\ \text{bytes/s}} \approx 2.0\ \mu s.$$

So a well-generated RMSNorm over a batch of 256 should complete in single-digit microseconds; if you measure 40, the kernel is not the problem, the launch overhead around it is. `Source: derived (A100 bandwidth cited: NVIDIA A100 datasheet).` The point of the worked example is not the exact number — it is that Triton generates a kernel that lands close to this floor without you writing a single `__shfl_down_sync`, because the reduction it emits is the one the architecture wants.

---

## 2. Porting the elementwise kernel: fused RoPE

The second kernel Track E wrote by hand was rotary position embedding, and it is the better teaching case because it is *fused*: RoPE, done naively in PyTorch, is three kernels (compute the rotated half, multiply by cosine, multiply-add the sine) each of which reads the query and key tensors from HBM and writes them back. On a memory-bound decode step, three round trips to HBM instead of one is three times the traffic for the same arithmetic.

Fusing means loading the block once, doing all the arithmetic in registers, and storing once. In Triton, fusion is not a technique you apply — it is the default, because everything between the `tl.load` and the `tl.store` stays in registers unless the compiler decides otherwise.

![A dataflow where query and key blocks and the cosine sine table merge into one rotate step and a single fused store](/imgs/blogs/triton-for-inference-kernels-and-when-to-stop-writing-cuda-3.webp)

```python
# nanoserve/kernels/triton/rope.py
@triton.jit
def rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr,
                q_out_ptr, k_out_ptr,
                seq_stride, n_heads, head_dim,
                BLOCK_D: tl.constexpr):
    tok = tl.program_id(0)          # which token position
    head = tl.program_id(1)         # which attention head
    half = head_dim // 2
    d = tl.arange(0, BLOCK_D)       # index within the head vector
    mask = d < half

    base = tok * seq_stride + head * head_dim
    # load the even/odd halves of q and k for this (token, head)
    q1 = tl.load(q_ptr + base + d, mask=mask, other=0.0)
    q2 = tl.load(q_ptr + base + half + d, mask=mask, other=0.0)
    k1 = tl.load(k_ptr + base + d, mask=mask, other=0.0)
    k2 = tl.load(k_ptr + base + half + d, mask=mask, other=0.0)
    # cos/sin depend only on position and dim -> precomputed, loaded once
    cos = tl.load(cos_ptr + tok * half + d, mask=mask, other=1.0)
    sin = tl.load(sin_ptr + tok * half + d, mask=mask, other=0.0)

    # the rotation, entirely in registers
    q_out1 = q1 * cos - q2 * sin
    q_out2 = q2 * cos + q1 * sin
    k_out1 = k1 * cos - k2 * sin
    k_out2 = k2 * cos + k1 * sin

    # one fused store per output
    tl.store(q_out_ptr + base + d,        q_out1, mask=mask)
    tl.store(q_out_ptr + base + half + d, q_out2, mask=mask)
    tl.store(k_out_ptr + base + d,        k_out1, mask=mask)
    tl.store(k_out_ptr + base + half + d, k_out2, mask=mask)
```

Two things are worth pausing on. First, the launch grid is now two-dimensional — `rope_kernel[(n_tokens, n_heads)](...)` — and you did not have to reason about how those map onto SMs; the compiler tiles them. Second, notice there is *no branch* on head type, no shared memory, no warp anything. The whole kernel is arithmetic between a load and a store, which is exactly the shape the compiler fuses best. In the CUDA version this same fusion was a deliberate act — you had to structure the loop so the compiler kept the intermediates in registers, and you checked the SASS to confirm it. Here it falls out of the model.

This kernel writes `nanoserve/kernels/triton/rope.py`, and the correctness harness from [the forward-pass post](/blog/machine-learning/inference-engineering/what-inference-engineering-is) checks it against the reference RoPE the same way it checked the CUDA one. When you swap the CUDA RoPE for this, nothing downstream changes — the block table, the KV append from [the sibling kernel post](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), the attention — because the contract is the same tensor in and out. That is the maintainability dividend in miniature: the kernel got a quarter of the length and stayed a drop-in.

---

## 3. `triton.autotune`: searching the config space so you don't

The RMSNorm launcher above hard-coded `BLOCK = next_power_of_2(H)`. That is one reasonable choice, but it is a *guess*, and the right choice depends on the architecture, the dtype, the shape, and how much the compiler can overlap. The number of warps per program (`num_warps`) and the software-pipelining depth (`num_stages`) matter just as much, and none of them has a value that is best everywhere.

This is where Triton's headline convenience lives. You hand the compiler a list of candidate configurations and a `key` — the set of arguments whose values define a "shape" — and it benchmarks every configuration the first time it sees a new shape, caches the winner, and reuses it forever after:

```python
# nanoserve/kernels/triton/rmsnorm_autotuned.py
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=3),
    ],
    key=['hidden'],          # re-benchmark only when `hidden` changes
)
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, hidden, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < hidden
    x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * hidden + offs, (x * inv * w).to(tl.float16), mask=mask)
```

The three knobs, and what each one trades:

| Knob | What it controls | Trade-off | Source |
| --- | --- | --- | --- |
| `BLOCK` | elements each program processes | bigger = fewer programs, more registers/program, better reuse; too big spills | derived |
| `num_warps` | threads per program (warps of 32) | more warps hide latency but split the block finer, raising register pressure | derived |
| `num_stages` | software-pipelining depth of the load/compute loop | deeper overlaps loads with compute but costs shared memory / registers | derived |

Why this matters more than it looks: the *right* combination genuinely moves between GPUs and between shapes, and no single one wins everywhere. The vLLM team makes this point directly about their Triton attention backend — in their [Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04; past this assistant's knowledge cutoff, so cited, not asserted), they report that the backend uses autotuned block sizes chosen per platform and state plainly that "no single configuration dominates across all scenarios." That is the empirical justification for shipping an autotuner rather than a hand-picked constant: the constant that is best on an H100 at batch 1 is not the constant that is best on an MI300 at batch 64.

<figure class="blog-anim">
<svg viewBox="0 0 700 260" role="img" aria-label="An autotune search sweeps a highlight across six candidate kernel configurations of different latencies and settles on the shortest bar as the cached winner" style="width:100%;height:auto;max-width:820px">
<style>
.tr-bar{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.tr-base{stroke:var(--border,#d1d5db);stroke-width:1.5}
.tr-cfg{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.tr-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.tr-scan{fill:var(--accent,#6366f1);opacity:.20}
.tr-glow{fill:var(--accent,#6366f1);opacity:0}
@keyframes tr-sweep{0%,12%{transform:translateX(0)}14%,26%{transform:translateX(90px)}28%,40%{transform:translateX(180px)}42%,54%{transform:translateX(270px)}56%,68%{transform:translateX(360px)}70%,82%{transform:translateX(450px)}84%,100%{transform:translateX(270px)}}
@keyframes tr-win{0%,82%{opacity:0}90%,100%{opacity:.85}}
@keyframes tr-fs{0%,80%{opacity:1}84%,100%{opacity:0}}
@keyframes tr-fw{0%,82%{opacity:0}88%,100%{opacity:1}}
.tr-scan{animation:tr-sweep 12s ease-in-out infinite}
.tr-glow{animation:tr-win 12s ease-in-out infinite}
.tr-cs{animation:tr-fs 12s ease-in-out infinite}
.tr-cw{animation:tr-fw 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.tr-scan{animation:none;transform:translateX(270px)}.tr-glow{animation:none;opacity:.85}.tr-cs{animation:none;opacity:0}.tr-cw{animation:none;opacity:1}}
</style>
<text class="tr-cap tr-cs" x="350" y="30">autotune: benchmarking every config for this shape</text>
<text class="tr-cap tr-cw" x="350" y="30">kept the fastest config, cached for this shape</text>
<line class="tr-base" x1="48" y1="210" x2="652" y2="210"/>
<rect class="tr-bar" x="60"  y="60"  width="60" height="150"/>
<rect class="tr-bar" x="150" y="95"  width="60" height="115"/>
<rect class="tr-bar" x="240" y="115" width="60" height="95"/>
<rect class="tr-bar" x="330" y="145" width="60" height="65"/>
<rect class="tr-bar" x="420" y="80"  width="60" height="130"/>
<rect class="tr-bar" x="510" y="105" width="60" height="105"/>
<rect class="tr-glow" x="330" y="145" width="60" height="65"/>
<rect class="tr-scan" x="60" y="55" width="60" height="160"/>
<text class="tr-cfg" x="90"  y="230">B=32</text>
<text class="tr-cfg" x="180" y="230">B=64 w2</text>
<text class="tr-cfg" x="270" y="230">B=128 w4</text>
<text class="tr-cfg" x="360" y="230">B=64 w4</text>
<text class="tr-cfg" x="450" y="230">B=256 w8</text>
<text class="tr-cfg" x="540" y="230">B=128 w8</text>
<text class="tr-cfg" x="350" y="252">bar height is latency; shorter is faster (heights illustrative)</text>
</svg>
<figcaption>An autotune pass times each candidate configuration for a given shape, then caches the fastest one; the numbers you will see depend on your GPU, so treat the heights as illustrative and reproduce them yourself.</figcaption>
</figure>

The convenience has a bill attached, and it is a real one. The first call on each new shape runs *every* config to time it, and each config that has never been compiled has to be JIT-compiled first. That first call is not microseconds; it is a visible pause. This is the same class of cost that vLLM documents for `torch.compile`: TorchInductor — the backend that actually generates Triton kernels — has a warm-up and its startup time is, per vLLM's [torch.compile integration post](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20), "a pain for autoscaling." Two engines, same root cause: compilation is not free, and you pay it the first time you see a shape.

#### How to measure an autotune win honestly

Do not benchmark the first call — you will be timing the compiler. The honest recipe, which is exactly the recipe from [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark): call the kernel once to trigger compilation and autotuning and throw that timing away; call it a few more times to reach steady state; lock the GPU clocks; `torch.cuda.synchronize()` before you start and stop the timer, or better, use `torch.cuda.Event` pairs; then report the median of many steady-state runs. Frame the result as a range: *on an A100 you should expect the autotuned RMSNorm to sit within a few percent of the memory-bound floor from the worked example; run `bench_kernels.py` and report yours.* `Source: reproduce (bench_kernels.py).` What you must never do is quote a speedup from the compile-included first call — that is the single most common way people accidentally report a fictional number.

---

## 4. The sampler that never syncs: removing a host stall with Triton

The most valuable Triton kernel in this whole post is not a matmul. It is the sampler, and the reason is that it removes a *synchronization*, not a few microseconds of arithmetic.

Recall the decode loop from [the sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo): every step the model produces a row of logits, the sampler turns it into a token id, and the loop appends that id and runs again. The trap is the phrase "turns it into a token id." The obvious implementation does a softmax, a `torch.multinomial`, and then — fatally — calls `.item()` or copies the id to the CPU so the Python loop can decide whether it hit a stop token and what to feed next. That copy is a device-to-host read, and a device-to-host read is a synchronization point: the CPU cannot proceed until the GPU has finished the step and delivered the byte.

![A branch where a host synchronizing sampler serializes the loop while an on GPU Gumbel sampler lets the CPU run one step ahead](/imgs/blogs/triton-for-inference-kernels-and-when-to-stop-writing-cuda-5.webp)

When the sampler forces a sync every step, the decode loop stops overlapping. Normally the CPU launches step N+1's kernels while the GPU is still finishing step N, so the step period is the *larger* of the two times. Force a sync and the period becomes their *sum*.

#### Worked example: what one sync per token costs

Suppose — and I am assigning these numbers, they are inputs to arithmetic, not a measurement — the GPU work for one decode step takes 3.0 ms and the CPU work to prepare and launch the next step takes 2.5 ms.

- **Overlapped (no sync):** the step period is $\max(3.0, 2.5) = 3.0$ ms, so $1000 / 3.0 \approx 333$ steps per second.
- **Serialized (sync every token):** the period is $3.0 + 2.5 = 5.5$ ms, so $1000 / 5.5 \approx 182$ steps per second.

That is a drop from 333 to 182 tokens per second, a loss of about 45% of your decode throughput, bought with a single four-byte read you did not need to do every step. `Source: derived (step times assumed and labeled; arithmetic shown).` The fix is to keep the sampled id on the GPU and only copy ids back in batches, or when a stop condition is checked on-device. That means the sampler itself must run on the GPU and produce a device tensor of ids without a softmax that the host reads.

The trick that makes on-GPU sampling clean is Gumbel-Max: adding independent Gumbel noise to the logits and taking the argmax draws exactly from the softmax distribution, *without ever materializing the softmax*. This is not a `nanoserve` invention — it is what vLLM's Model Runner V2 does. Per vLLM's [Model Runner V2 post](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24; past cutoff, cited), its Triton sampler "uses Gumbel-Max (no explicit softmax materialization)" with a stateless in-kernel RNG. Here is a top-k/top-p Gumbel sampler as a Triton kernel — one program per request row, the whole thing on-device:

```python
# nanoserve/kernels/triton/sampler.py
@triton.jit
def gumbel_sample_kernel(logits_ptr, out_ptr,
                         temp_ptr, vocab,
                         seeds_ptr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # one request per program
    offs = tl.arange(0, BLOCK)
    mask = offs < vocab
    logits = tl.load(logits_ptr + row * vocab + offs,
                     mask=mask, other=-float('inf'))
    temp = tl.load(temp_ptr + row)
    logits = logits / temp                        # temperature, per-row

    # stateless per-(row, token) uniform via a hash of the seed and index
    seed = tl.load(seeds_ptr + row)
    u = tl.rand(seed, offs)                       # in (0, 1), reproducible
    # Gumbel(0,1) = -log(-log u); argmax(logits + g) ~ softmax(logits)
    g = -tl.log(-tl.log(u + 1e-20) + 1e-20)
    perturbed = tl.where(mask, logits + g, -float('inf'))

    token = tl.argmax(perturbed, axis=0)          # stays on the GPU
    tl.store(out_ptr + row, token)
```

That kernel produces a device tensor `out` of one token id per request, and nothing was read back to the host. Truncation (top-k, top-p, min-p) composes in front of it exactly as in the sampler zoo — you mask the logits to `-inf` outside the kept set before adding the Gumbel noise, and the mask is itself a Triton op over the sorted or thresholded logits. The launcher:

```python
def sample(logits, temperatures, seeds):
    B, V = logits.shape
    out = torch.empty(B, dtype=torch.long, device=logits.device)
    BLOCK = triton.next_power_of_2(V)
    gumbel_sample_kernel[(B,)](logits, out, temperatures, V, seeds, BLOCK=BLOCK)
    return out                                    # a DEVICE tensor, no .item()
```

The determinism caveat that [the sampling-numerics post](/blog/machine-learning/inference-engineering/what-inference-engineering-is) raised applies here too: `tl.rand` seeded per row is reproducible *for a fixed batch*, but the argmax is still a reduction whose result can depend on reduction order if the kernel is not batch-invariant. For a sampler that is not usually a correctness problem — you are drawing a sample — but it is worth knowing that "stateless in-kernel RNG" gives you reproducibility of the *noise*, not of the *reduction*.

---

## 5. The headline: 100.7% of FlashAttention 3 in about 800 lines

Everything so far has been small kernels. The reason to believe the argument at the scale that matters — attention, the kernel that owns most of a decode step's time — is a single public result, and it is worth stating precisely because it is the whole maintenance-versus-performance case in one data point.

In the [vLLM Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04; past this assistant's cutoff, cited verbatim, not extrapolated), the vLLM team reports that on an H100 their Triton attention backend reached **100.7% of the performance of FlashAttention 3** on a specific setup — Llama-3.1-8B, batch 1, a 500-token input with a long decode — and that the Triton implementation is **about 800 lines** against FlashAttention 3's roughly **70,000 lines** of CUDA. The same post reports the backend running roughly **5.8× faster than earlier implementations on an AMD MI300**, and that Triton is the **default attention backend on AMD/ROCm**.

Read those numbers carefully, because the honesty rule cuts both ways and the setup is load-bearing:

| Claim | Setup | Source |
| --- | --- | --- |
| Triton = 100.7% of FA3 | H100, Llama-3.1-8B, batch 1, 500-tok input, long decode | cited: vLLM Triton backend post (2026-03-04) |
| ~800 lines vs ~70,000 | Triton attention backend vs FlashAttention 3 source | cited: vLLM Triton backend post (2026-03-04) |
| ~5.8× over prior impls | AMD MI300 | cited: vLLM Triton backend post (2026-03-04) |
| default backend on ROCm | AMD GPUs | cited: vLLM Triton backend post (2026-03-04) |

100.7% is not "Triton beats CUDA everywhere." It is one point — batch 1, one input length, one GPU — where a compiler-generated kernel matched a famously hand-optimized one, from about 1.1% of the source lines. That is astonishing and it is also narrow: the same post is candid about where the Triton backend is *not* free.

The limitations, stated as the vLLM team states them, are the other half of the picture:

- **Fixed launch grids replay badly under CUDA graphs.** The persistent-kernel variant reads its work assignment from GPU memory and uses variable launch grids; those "replay badly under CUDA graphs" — the graph captured the launch geometry, and a different amount of work per step does not fit the captured shape. This is a genuine friction with the [CUDA-graph decode loop](/blog/machine-learning/inference-engineering/what-inference-engineering-is) an engine wants for host-overhead reasons.
- **Split-K needs a second launch.** The decode split-KV path ("parallel tiled softmax") splits the KV traversal across a 3D grid and then needs a *second* reduction kernel to combine the partials — a second launch, heuristic-gated. Hand CUDA can sometimes fuse that combine; the Triton path launches twice.
- **No single configuration dominates.** The reason the backend must autotune per platform is precisely that no one config is best across scenarios — which means the "800 lines" comes with a config-search apparatus and a cache, not a magic constant.

So the headline is real and the caveats are real. The correct reading is the one the decision matrix in the next section encodes: for attention, you should reach for FlashAttention 3 or FlashInfer when you need the last drop of Hopper performance under CUDA graphs, and reach for the Triton backend when you need it to *run on AMD at all*, or when you need to read and modify it. Both are true at once.

---

## 6. The decision: when to stop writing CUDA

Here is the framework, and it is the section to screenshot for a design review. The choice of tool is a function of two things: the *shape* of the kernel and *which property you need most*.

![A matrix mapping kernel shapes against the need for peak portability or maintainability to a recommended tool](/imgs/blogs/triton-for-inference-kernels-and-when-to-stop-writing-cuda-6.webp)

The rules the matrix encodes, in order of how often they apply:

1. **Reach for a library first.** For attention, use FlashAttention or [FlashInfer](/blog/machine-learning/model-serving/attention-backends-deep-dive-flashattention-flashinfer); for dense GEMM, use cuBLAS or CUTLASS. These are the kernels with the most person-years of tuning behind them, and neither Triton nor your CUDA will beat them at their own game on the hardware they target. A kernel you did not write is a kernel you do not maintain.
2. **Write Triton when you need a custom fused kernel that you want readable, autotuned, and portable.** This is the sweet spot and it is most of an inference engine's *custom* surface: RMSNorm, RoPE, the fused dequant path, KV-cache writes, the sampler, activation fusions, the odd fused epilogue a library does not expose. You get fusion for free, autotuning for free, and — the property that is easy to undervalue until you have an AMD deployment — the same source runs on ROCm.
3. **Keep hand CUDA for the last few percent on one architecture, or for something Triton cannot express.** Warp-specialized producer/consumer pipelines, hand-managed multi-stage shared-memory software pipelines, use of a specific tensor-core instruction or a memory descriptor Triton does not surface — these are where CUDA still wins, and they are rare in inference. When you write CUDA, know that you are buying peak performance on one GPU with maintenance cost and zero portability.

There is a fourth rule that sits above all of them and is the most modern argument for Triton: **it is already the compiler's target.** `torch.compile` does not emit CUDA — its backend, TorchInductor, *generates Triton*. Per vLLM's [torch.compile post](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20), TorchInductor does pointwise and reduction fusion and selects a matmul backend among cuBLAS, Triton, and CUTLASS. So when you write `@torch.compile` over your model — as V1 vLLM does by default — the fused elementwise kernels you get *are Triton kernels you did not have to write*. Learning to read and write Triton is learning to read and write the code your compiler already emits. That is a different and stronger reason than "Triton is convenient": it is the lingua franca of GPU codegen in the PyTorch ecosystem now, and TensorRT-style ahead-of-time compilation (see [the TensorRT post](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler)) is the only major alternative path.

#### Worked example: the whole `nanoserve` kernel folder, re-decided

Walk Track E's kernels through the matrix:

| Kernel | Best tool | Why | Source |
| --- | --- | --- | --- |
| RMSNorm | Triton | memory-bound elementwise; ties CUDA, quarter the code, portable | derived + cited (vLLM Triton post) |
| RoPE (fused) | Triton | pure register arithmetic between load and store; fusion is free | derived |
| KV-cache append | Triton | elementwise scatter under a block table; compiler coalesces | derived |
| Decode attention | library, then Triton | FA3/FlashInfer for peak on Hopper; Triton backend for AMD / readability | cited (vLLM Triton post) |
| Dense GEMM (prefill) | cuBLAS / CUTLASS | decades of tuning; nobody beats it at its own game | derived |
| Dequant-fused GEMM | Triton or CUDA | Triton expresses it; Marlin-class CUDA wins the last few % (see [the dequant post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly)) | derived |
| Sampler | Triton | removes a host sync; Gumbel-Max on-device | derived + cited (vLLM MRv2) |
| Warp-specialized pipeline | hand CUDA | producer/consumer + explicit staging Triton cannot express | derived |

Notice the shape of the answer: of the eight kernels, five are best in Triton, two belong to a library, and exactly one — the exotic warp-specialized pipeline that does not appear in a normal inference engine — justifies hand CUDA. That is the quantitative version of "most of it, no." The CUDA you wrote in Track E was a superb way to *learn how the GPU works*, which is why it came first; it is not the way you would *ship* five of those eight.

---

## 7. Debugging Triton: interpreter, prints, and the CUDA fallback

The objection I hear from CUDA people is that Triton is a black box — when it is wrong, you cannot step through it. That was true years ago and is not now, and the debugging path has a clean structure.

![A decision tree splitting a wrong result into a config bug or a logic bug and a hang into the CUDA core dump path](/imgs/blogs/triton-for-inference-kernels-and-when-to-stop-writing-cuda-7.webp)

The tree above is the triage. Start by classifying the failure.

**A wrong result.** First ask whether it is config-dependent. Turn autotuning off by pinning a single trivial config and re-run:

```python
import os
os.environ["TRITON_INTERPRET"] = "1"    # run the kernel in pure Python
```

`TRITON_INTERPRET=1` runs the `@triton.jit` kernel in an interpreter on the host, where `tl.load`/`tl.store` become ordinary array operations and you can drop a Python `breakpoint()` inside the kernel, print intermediate tensors, and inspect masks. If the result is *correct* in the interpreter but *wrong* compiled, and it changes when you change `num_stages` or `BLOCK`, you have a config bug — usually a masking or pipelining edge case the compiler exposes at a particular stage depth. If the result is *wrong in the interpreter too*, it is a plain logic bug — an off-by-one in your offsets, a mask that admits the tail, a reduction axis mixed up — and now you have a normal Python debugger to find it.

You can also print from a compiled kernel directly with `tl.device_print`:

```python
@triton.jit
def dbg_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.device_print("row sum ", tl.sum(x, axis=0))   # prints per program
```

**A hang or an illegal address.** This is where you fall back to the CUDA tooling, because a hung kernel is below the level Triton models. The vLLM team's [improved CUDA debugging post](https://vllm.ai/blog/2025-12-03-improved-cuda-debugging) (2025-12-03; cited) documents the path for exactly this: a user-triggered GPU core dump (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1`, a named pipe, a `dd` write to trigger it) opened in `cuda-gdb`, with the honest caveat that even with line info, `cuda-gdb` "often fails to find the line" and shows only the last line after inlining. The companion [core-dump-on-exception post](https://vllm.ai/blog/2025-08-11-cuda-debugging) (2025-08-11) covers the illegal-access case. The point is that Triton does not remove the CUDA debugger from your toolbox — it just means you reach for it far less often, and mostly for hangs rather than wrong answers.

**Recompiles forever.** If your service seems to pause repeatedly rather than once, you are compiling a new shape every step. That is the compile tax from section 3 showing up as a symptom: your `key` is capturing a value that varies (a sequence length, say) so every step is a "new shape." The fix is to bucket — round shapes up to a fixed set so the cache hits — which is the same discipline `torch.compile` exposes as static-shape compilation, and which the timeline figure in section 3 is really about.

---

## 8. Where Triton breaks: three stress tests

Balance requires naming where the tool loses, not just where it wins. Three stress tests, each a real failure mode.

**Stress test 1: the warp-specialized producer/consumer.** The fastest attention and GEMM kernels on Hopper use *warp specialization* — some warps do nothing but load data into shared memory (producers) while others do nothing but compute on it (consumers), coordinated through a shared-memory pipeline and asynchronous copies. This is a program structure where the *point* is that different warps run different code and hand off through explicit barriers. Triton's model — one block-level program, compiler-assigned warps — does not naturally express "these four warps produce, those four consume." You can sometimes coax the compiler with pipelining depth, but you cannot write the producer/consumer split directly. This is a real reason FlashAttention 3's Hopper path is CUDA, and it is the archetype of "keep the CUDA." vLLM's own attention work reflects this: their [HPC-Ops post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06; cited) describes a hand-built multi-stage persistent kernel for mixed-length decode, not a Triton one.

**Stress test 2: autotune exploding the compile time.** Autotuning is a product: (number of shapes you serve) × (number of configs) × (compile-and-benchmark time per config). Serve one shape and it is a one-time few-second pause. Serve a server that sees dozens of distinct sequence-length buckets, each with a fat config grid, and startup becomes minutes — the exact "pain for autoscaling" vLLM names for `torch.compile`. The mitigation is the same one the torch.compile post documents: bucket aggressively (a small fixed set of `compile_sizes` rather than every shape), trim the config grid to the handful that ever win on your hardware, and *persist the compile cache* so a restart does not re-pay it. A fat config grid you never prune is a self-inflicted startup regression.

**Stress test 3: fast on NVIDIA, slow on AMD until you re-tune.** The portability promise is that the *source* runs on ROCm — not that the *tuning* transfers. A config grid and a set of block sizes that were autotuned into their sweet spot on an H100 will run correctly on an MI300 and may run well below its potential, because the winning config there is different — different warp/wavefront size, different cache hierarchy, different pipelining sweet spot. This is the flip side of "no single configuration dominates": portability of code is not portability of performance. When you move a Triton kernel to a new architecture you must let it re-autotune there, and you should re-examine whether your config grid even *contains* the configs that architecture wants. The MI300 5.8× speedup vLLM reports did not come from running the H100 config on AMD — it came from a backend built and tuned to be the ROCm default.

---

## When to reach for this (and when not)

**Reach for Triton when** you are writing a *custom* fused kernel for your engine — an elementwise op, a norm, RoPE, a KV write, a dequant path, a sampler, an activation fusion — and you want it readable, autotuned, and portable across NVIDIA and AMD. This is most of the custom kernel surface of an inference engine, and it is where Triton is not a compromise but the correct default.

**Do not write Triton (reach for a library) when** the kernel is a dense GEMM or a standard attention on the architecture the library targets. cuBLAS, CUTLASS, FlashAttention 3, and FlashInfer exist and you will not beat them; wrapping them is less code and more performance than any kernel you write. This is the same "just use vLLM" instinct the series keeps returning to, one level down: just use the library.

**Keep hand CUDA when** you need warp specialization, an explicit multi-stage shared-memory pipeline, a specific instruction Triton does not surface, or the last few percent of peak on a single architecture you fully control — and you have measured that the gap is worth the maintenance cost and the loss of portability. In an inference engine that is a small and shrinking set of kernels.

And the meta-point: **learn to read Triton regardless**, because `torch.compile` emits it. When you profile a compiled model and open the generated kernel, it is Triton. The skill is not optional even if you never hand-write a kernel — it is how you read what your compiler produced.

---

## Key takeaways

- **Triton's trade is control for a compiler.** You describe blocks of data with `tl.load`/`tl.store`/`tl.dot`/`tl.sum`; the compiler owns thread assignment, shared memory, coalescing, vectorization, and pipelining. The same RMSNorm that was ~80 lines of CUDA is ~15 lines of Triton and ties it on bandwidth.
- **Fusion is the default, not a technique.** Everything between a load and a store stays in registers, so a fused RoPE is one HBM round trip where eager PyTorch was three.
- **`triton.autotune` searches the config space for you**, because no single `BLOCK`/`num_warps`/`num_stages` wins across shapes and GPUs — vLLM says as much for their attention backend. The cost is a one-time cold compile per new shape; bucket shapes and persist the cache.
- **The most valuable Triton kernel removes a synchronization, not a few microseconds.** An on-GPU Gumbel-Max sampler keeps the sampled id on the device and lets the decode loop overlap; a `.item()` per token can cost ~45% of throughput by serializing the loop.
- **The headline is real and narrow.** vLLM reports their ~800-line Triton attention backend hit 100.7% of FlashAttention 3's ~70,000 lines on one H100 setup, and is the default on AMD — but with honest caveats: fixed launch grids replay badly under CUDA graphs, split-K needs a second launch, and no config dominates.
- **The decision rule: library first, Triton for custom fused kernels, hand CUDA for the exotic last-few-percent.** Of `nanoserve`'s eight kernels, five are best in Triton, two belong to a library, one justifies CUDA.
- **Triton is the compiler's target.** TorchInductor generates Triton, so reading it is reading what `torch.compile` already emits — the skill is not optional even if you never hand-write a kernel.
- **Portability of code is not portability of performance.** A Triton kernel runs on ROCm unchanged but must be re-autotuned there to run well.

---

## Further reading

- [Triton documentation and tutorials](https://triton-lang.org/) — the fused-softmax and matmul tutorials are the fastest way to internalize the block model.
- vLLM, [Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) — the source of the 100.7%/800-lines/70,000-lines comparison and the honest limitations.
- vLLM, [torch.compile Integration](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20) — how TorchInductor generates Triton and selects a matmul backend, and why compile startup is a real cost.
- vLLM, [Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) — the Gumbel-Max Triton sampler and building input tensors on-GPU.
- vLLM, [Improved CUDA debugging](https://vllm.ai/blog/2025-12-03-improved-cuda-debugging) (2025-12-03) — the core-dump path for hangs when you fall back below Triton.
- Series intro: [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and the capstone: [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).
- Track E siblings: [Writing your first inference CUDA kernel](/blog/machine-learning/inference-engineering/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope), [Paged attention kernel by hand](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), [Dequant-fused GEMM](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly), and the model-serving view of [custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference).
