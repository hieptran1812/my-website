---
title: "Writing Your First Inference CUDA Kernel: RMSNorm and RoPE"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Write two real, compilable CUDA kernels for LLM inference — a block-reduction RMSNorm and a rotation-based RoPE — then fuse them into a single-pass prologue, prove it correct against the reference, and debug the segfault you will inevitably hit."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "cuda",
    "kernels",
    "rmsnorm",
    "rope",
    "pytorch",
    "gpu",
    "ml-systems",
    "kernel-fusion",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

The [kernel landscape post](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs) ended with a list: for one decode step of Llama-3.1-8B, the profiler shows roughly ten kernels, and the ones that surprise people are not the matmuls. They are the small ones — the RMSNorms, the rotary embedding, the residual adds — that each read the residual stream out of high-bandwidth memory (HBM, the ~1–3 TB/s DRAM sitting next to the GPU cores), do a trivial amount of arithmetic, and write it straight back. There are 64 RMSNorm launches per token for a 32-layer model. Each one is memory-bound, each one is a separate kernel launch, and each one round-trips a vector that the *next* kernel is about to read again.

That is the whole opportunity in this post. RMSNorm and RoPE are the perfect first CUDA kernels precisely because they are boring: no matmul, no tensor cores, no online softmax. But they are not *trivial*. RMSNorm needs a **reduction** — every thread in a block has to agree on one sum-of-squares — and RoPE needs a **rotation** with a layout convention that is the single most common source of subtly-wrong logits in hand-written inference code. Get those two patterns into your hands and you have the vocabulary for every kernel later in this track: the KV-cache append, the paged-attention decode, the fused sampler. Figure 1 shows where these ops sit in a layer's dataflow and why the interesting fusion is not the one you would guess.

By the end you will have written `nanoserve/kernels/prologue.cu` — a real, compilable CUDA extension loaded through `torch.utils.cpp_extension.load` — containing a warp-shuffle RMSNorm, a coalesced RoPE, and a single fused kernel that reads the query and key tensors once and writes them once, saving two HBM round-trips. You will have a parity harness that diffs it against the pure-PyTorch reference from [the forward-pass post](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch), and you will know exactly which environment variables to set when it segfaults, because your first CUDA kernel *will* segfault. This is a companion to the HPC series' [first-kernel walkthrough](/blog/machine-learning/high-performance-computing/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel); that post teaches CUDA from zero, this one teaches the two kernels an inference engine actually launches.

![Dataflow of one layer prologue where the residual stream feeds RMSNorm then the QKV projection which branches into per-head rotary embedding on queries and keys before merging into attention](/imgs/blogs/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope-1.webp)

The figure makes the key structural point up front: the pre-attention RMSNorm and the RoPE are *not adjacent*. A GEMM — the QKV projection — sits between them. So the fusion everyone reaches for first ("fuse the norm and the rope") is impossible without also fusing the matmul, which is a different and much harder kernel. The fusion that *is* possible, and that production engines actually ship, sits entirely on the far side of the projection: the per-head normalization of Q and K (when the model has it), the rotary embedding, and the KV-cache write, all elementwise on the projected q/k tensors. We will build up to exactly that.

## 1. The execution model you actually need

You do not need a semester of CUDA to write these two kernels. You need five ideas, and this section is all five. For the full treatment — occupancy, the memory hierarchy, launch configuration — read the [CUDA-for-AI-engineers post](/blog/machine-learning/high-performance-computing/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel) and the [memory-hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm); here I compress it to what these kernels touch.

**A kernel is one function that every thread runs, differing only by its index.** You do not write a loop over tokens. You launch thousands of threads, and each one computes its own coordinates from built-in variables (`blockIdx`, `threadIdx`) and does the slice of work those coordinates name. The "for loop" is the hardware launching the threads.

**Threads are grouped into blocks; blocks form a grid.** A block is a team of threads (up to 1024) that share fast on-chip **shared memory** and can synchronize with a barrier (`__syncthreads()`). Blocks cannot cheaply talk to each other, so the first design decision for any kernel is *what does one block own?* For RMSNorm the answer writes itself: **one block per token (per row)**. A token's hidden vector is 4096 numbers that all need to see the same sum-of-squares — that is exactly a block-sized cooperative job. Different tokens are independent, so they get different blocks.

**Within a block, threads run in warps of 32.** A warp is the real unit of execution: 32 threads that issue the same instruction together (SIMT — single instruction, multiple threads). This matters twice in these kernels. First, threads in a warp can exchange registers directly with `__shfl_*` instructions, no shared memory needed — that is the fast reduction. Second, when a warp issues a memory load, the hardware wants those 32 addresses to fall in one contiguous 128-byte segment so it can serve them in one transaction. That is **coalescing**, and it is the difference between using your bandwidth and wasting seven-eighths of it.

![Layered hierarchy from the launch grid down through blocks warps and threads to the shared memory used for cross warp partial sums](/imgs/blogs/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope-2.webp)

**Shared memory is a small, fast scratchpad per block.** On an A100 each streaming multiprocessor has up to 164 KB of it, and it is roughly an order of magnitude lower latency than HBM. You use it to pass partial results *between* warps of the same block — because warps in a block do not otherwise share registers. It is the glue in a block-wide reduction.

**The launch decides the shape.** `kernel<<<grid, block>>>(args)` picks how many blocks and how many threads per block. For RMSNorm on a batch of `S` tokens with hidden size `H`, we launch `S` blocks of, say, 256 threads, and each thread strides over `H/256 = 16` of the 4096 elements. Those five ideas — kernel-as-per-thread-function, block-owns-a-token, warp-of-32, coalescing, shared-memory-glue — are the entire conceptual budget for what follows.

## 2. RMSNorm, the honest way and the wrong way

Recall the definition from [the forward-pass post](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch). RMSNorm drops LayerNorm's mean-subtraction and bias, keeping only the rescale:

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot g_i
$$

The pure-PyTorch reference we are going to match, verbatim from that post, is four lines:

```python
# nanoserve/model.py — the reference we must reproduce bit-for-bit-ish
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)   # mean of squares, fp32
    x = x * torch.rsqrt(variance + eps)          # eps INSIDE the sqrt
    return weight * x.to(in_dtype)               # cast back, THEN scale
```

Three details in those four lines decide whether your kernel matches, and every one of them is a decision you have to make explicitly in CUDA rather than inherit from a library:

- **The sum of squares is accumulated in fp32**, even when the input is bf16. This is not optional. bf16 carries eight significand bits, so its machine epsilon is about $2^{-7} \approx 0.008$; summing 4096 squared bf16 values in bf16 loses precision catastrophically, and the error is different from the reference's, so your parity check fails. This is the same numerics discipline the [sampling-numerics post](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance) insists on for logits: reduce in fp32, always.
- **Epsilon goes inside the square root**, added to the mean of squares. Outside is a different function that will pass casual eyeballing and fail a tight tolerance forever.
- **The cast back to bf16 happens before the weight multiply.** The reference computes `x.to(in_dtype)` first, rounding the normalized activation to bf16, and only then multiplies by the weight. Reorder that and you get a few units-in-the-last-place of drift — enough, occasionally, to flip a marginal argmax.

Here is the naive kernel — correct, readable, and deliberately using the *slow* reduction so we can replace it in the next section and see why:

```cuda
// nanoserve/kernels/prologue.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

// Naive block reduction via shared memory. One block owns one token row.
template <typename scalar_t>
__global__ void rmsnorm_naive_kernel(
    scalar_t* __restrict__ out,          // [S, H]
    const scalar_t* __restrict__ inp,    // [S, H]
    const scalar_t* __restrict__ weight, // [H]
    const float eps,
    const int H) {
  const int row = blockIdx.x;                       // which token
  const scalar_t* x = inp + (size_t)row * H;
  scalar_t* y = out + (size_t)row * H;

  // Each thread sums the squares of its strided slice, IN FP32.
  float partial = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = static_cast<float>(x[i]);
    partial += v * v;
  }

  // Slow tree reduction in shared memory: log2(blockDim) syncthreads.
  extern __shared__ float sdata[];
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();                                // barrier EVERY step
  }
  float ss = sdata[0];

  __shared__ float inv_rms;
  if (threadIdx.x == 0) inv_rms = rsqrtf(ss / H + eps);   // eps inside sqrt
  __syncthreads();

  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = static_cast<float>(x[i]) * inv_rms;
    scalar_t xn = static_cast<scalar_t>(v);               // cast back FIRST
    y[i] = static_cast<scalar_t>(static_cast<float>(xn)
                                 * static_cast<float>(weight[i]));
  }
}
```

Read the reduction loop carefully, because it is the thing we are about to beat. It halves the number of active threads each step and calls `__syncthreads()` after each halving. For a 256-thread block that is eight barriers, and — worse — the shared-memory accesses have bank conflicts and most threads sit idle for most of the steps. Every one of those barriers stalls the whole block. It works, it is correct, and on a memory-bound op the launch and reduction overhead is a real fraction of the runtime.

The `for (int i = threadIdx.x; i < H; i += blockDim.x)` pattern is a **grid-stride loop** over the row. Thread 0 handles element 0, 256, 512, …; thread 1 handles 1, 257, …. That striding is not arbitrary: in any single iteration, the 256 threads of the block read elements `[k, k+255]`, which are contiguous in memory, so the load coalesces. If instead you gave thread 0 the first 16 elements and thread 1 the next 16 (a "chunked" split), the 32 threads of a warp would read 32 addresses 16 elements apart, and the hardware would need many transactions instead of one. Same arithmetic, a fraction of the bandwidth. This is the single most important habit to build.

#### Worked example: how much traffic is one RMSNorm?

Llama-3.1-8B has $H = 4096$ and runs in bf16 (2 bytes). One RMSNorm reads the row and writes it back: $2 \cdot H \cdot 2 = 16{,}384$ bytes per token, or 16 KB. The weight vector is another $H \cdot 2 = 8$ KB, but it is the same 8 KB for every token, so it stays resident in L2 and the amortized cost per token is near zero. The model has two RMSNorms per layer plus one final norm: $2 \cdot 32 + 1 = 65$ invocations per token, about 1 MB of pure norm traffic per generated token. On an A100 at ~2 TB/s that is roughly half a microsecond of *bandwidth* — trivial. What is not trivial is 65 separate kernel launches, each with its own overhead, which is precisely why fusion and CUDA graphs exist. Source: derived from the config in the forward-pass post and the A100 bandwidth cited in section 8.

## 3. The warp-shuffle reduction, and why it wins

The naive reduction's problem is that it treats all 256 threads as strangers who can only communicate by writing to a shared bulletin board and waiting at a barrier. But threads within a warp are not strangers — they run in lockstep and can hand each other register values directly. `__shfl_down_sync` reads a register from another lane in the same warp with no shared memory and no barrier:

```cuda
// Reduce 32 lanes of a warp to a single sum in lane 0. Five instructions.
__inline__ __device__ float warp_reduce_sum(float val) {
  // full 32-lane mask; every active lane participates
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, offset);
  return val;                 // lane 0 holds the total
}
```

Five iterations — offsets 16, 8, 4, 2, 1 — collapse 32 partial sums into one, because $\log_2 32 = 5$. No `__syncthreads()`, no shared memory, no bank conflicts, and the whole thing lives in registers. The animation below walks those five steps: on each step the number of lanes still carrying live data halves, and after step five a single lane holds the sum.

<figure class="blog-anim">
<svg viewBox="0 0 680 300" role="img" aria-label="A warp-shuffle reduction collapsing thirty-two partial sums to one across five halving steps, with a highlight sweeping down the five levels" style="width:100%;height:auto;max-width:760px">
<style>
.ws-bar{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ws-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:start}
.ws-num{font:600 14px ui-monospace,ui-sans-serif;fill:var(--text-secondary,#6b7280);text-anchor:end}
.ws-hi{fill:none;stroke:var(--accent,#6366f1);stroke-width:3.5;rx:8}
@keyframes ws-drop{0%{transform:translateY(0)}12%{transform:translateY(0)}20%{transform:translateY(44px)}32%{transform:translateY(44px)}40%{transform:translateY(88px)}52%{transform:translateY(88px)}60%{transform:translateY(132px)}72%{transform:translateY(132px)}80%{transform:translateY(176px)}92%{transform:translateY(176px)}100%{transform:translateY(220px)}}
.ws-sweep{animation:ws-drop 11s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ws-sweep{animation:none}}
</style>
<rect class="ws-bar" x="40" y="20"  width="560" height="34" rx="8"/>
<rect class="ws-bar" x="40" y="64"  width="280" height="34" rx="8"/>
<rect class="ws-bar" x="40" y="108" width="140" height="34" rx="8"/>
<rect class="ws-bar" x="40" y="152" width="70"  height="34" rx="8"/>
<rect class="ws-bar" x="40" y="196" width="35"  height="34" rx="8"/>
<rect class="ws-bar" x="40" y="240" width="18"  height="34" rx="8"/>
<text class="ws-num" x="34" y="42">32</text>
<text class="ws-num" x="34" y="86">16</text>
<text class="ws-num" x="34" y="130">8</text>
<text class="ws-num" x="34" y="174">4</text>
<text class="ws-num" x="34" y="218">2</text>
<text class="ws-num" x="34" y="262">1</text>
<text class="ws-lbl" x="616" y="42">start: 32 partial sums</text>
<text class="ws-lbl" x="330" y="86">shfl offset 16</text>
<text class="ws-lbl" x="190" y="130">shfl offset 8</text>
<text class="ws-lbl" x="120" y="174">shfl offset 4</text>
<text class="ws-lbl" x="85" y="218">shfl offset 2</text>
<text class="ws-lbl" x="68" y="262">shfl offset 1 - done</text>
<rect class="ws-hi ws-sweep" x="36" y="16" width="568" height="42"/>
</svg>
<figcaption>Each shuffle step folds every lane into the one 16, then 8, then 4, then 2, then 1 position below it, so the live data halves five times and a single lane ends holding the warp total.</figcaption>
</figure>

To reduce a whole 256-thread block, you compose: each of the 8 warps reduces itself with `warp_reduce_sum`, each warp's lane 0 writes its result into an 8-slot shared-memory array, and then the *first* warp reduces those 8 values with one more `warp_reduce_sum`. Two warp reductions and a single barrier, versus the naive version's eight barriers:

```cuda
__inline__ __device__ float block_reduce_sum(float val) {
  static __shared__ float warp_sums[32];   // one slot per warp, max 1024/32
  const int lane = threadIdx.x & 31;        // threadIdx.x % 32
  const int wid  = threadIdx.x >> 5;        // threadIdx.x / 32
  val = warp_reduce_sum(val);               // reduce within each warp
  if (lane == 0) warp_sums[wid] = val;      // one value per warp -> shared
  __syncthreads();                          // the ONLY barrier
  const int n_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < n_warps) ? warp_sums[lane] : 0.0f;
  if (wid == 0) val = warp_reduce_sum(val); // first warp reduces the warp sums
  return val;                               // lane 0 of warp 0 holds the total
}
```

Now the RMSNorm kernel is the same as before with the slow loop swapped out, and it needs no dynamic shared memory:

```cuda
template <typename scalar_t>
__global__ void rmsnorm_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ inp,
    const scalar_t* __restrict__ weight, const float eps, const int H) {
  const int row = blockIdx.x;
  const scalar_t* x = inp + (size_t)row * H;
  scalar_t* y = out + (size_t)row * H;

  float partial = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = static_cast<float>(x[i]);
    partial += v * v;
  }
  float ss = block_reduce_sum(partial);

  __shared__ float inv_rms;
  if (threadIdx.x == 0) inv_rms = rsqrtf(ss / H + eps);
  __syncthreads();

  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    scalar_t xn = static_cast<scalar_t>(static_cast<float>(x[i]) * inv_rms);
    y[i] = static_cast<scalar_t>(static_cast<float>(xn)
                                 * static_cast<float>(weight[i]));
  }
}
```

![Ordered sequence of the six stages inside the RMSNorm kernel from loading and promoting to fp32 through the warp shuffle and block reduction to the scale and store step](/imgs/blogs/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope-3.webp)

Why does the warp shuffle actually win, beyond aesthetics? Three reasons, all of which matter more on a memory-bound op where the kernel is short and overhead is a large fraction of the total. It removes seven of the eight barriers, and a barrier stalls every warp in the block until the slowest arrives. It never touches shared memory for the intra-warp part, so there are no bank conflicts and no shared-memory latency on the hot path. And it keeps the reduction operands in registers, which is where you want them. The canonical reference for this pattern is NVIDIA's developer post ["Faster Parallel Reductions on Kepler"](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) by Justin Luitjens, which is where `__shfl_down` reductions entered the mainstream. The magnitude of the win is workload- and hardware-dependent, so treat it as a pattern to adopt rather than a fixed speedup you can quote — measure it yourself with the harness in section 6.

## 4. Loading the kernel and wiring the launch

Before RoPE, let us make what we have runnable, because a kernel you cannot launch is a kernel you cannot debug. PyTorch's `torch.utils.cpp_extension.load` compiles a `.cu` file at import time and hands you back a Python module. The C++ side needs a launcher that checks its inputs, computes the grid, dispatches on dtype, and binds to Python:

```cpp
// nanoserve/kernels/prologue.cu (continued)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor rmsnorm(torch::Tensor inp, torch::Tensor weight, double eps) {
  CHECK_CUDA(inp); CHECK_CUDA(weight);
  CHECK_CONTIG(inp); CHECK_CONTIG(weight);
  TORCH_CHECK(inp.dim() == 2, "inp must be [S, H]");
  const int S = inp.size(0), H = inp.size(1);
  TORCH_CHECK(weight.size(0) == H, "weight must be [H]");

  auto out = torch::empty_like(inp);
  const int threads = 256;                 // multiple of warp size
  const dim3 grid(S), block(threads);      // one block per token row

  AT_DISPATCH_REDUCED_FLOATING_TYPES(inp.scalar_type(), "rmsnorm", [&] {
    rmsnorm_kernel<scalar_t><<<grid, block>>>(
        out.data_ptr<scalar_t>(), inp.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(), (float)eps, H);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();          // surfaces launch errors now
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "Fused RMSNorm (CUDA)");
}
```

`AT_DISPATCH_REDUCED_FLOATING_TYPES` expands the templated kernel for `at::Half` and `at::BFloat16` and picks the right instantiation at runtime — it is how one `.cu` serves both fp16 and bf16 without you writing the switch. `C10_CUDA_KERNEL_LAUNCH_CHECK()` is the line beginners skip and then spend an afternoon regretting: kernel launches are asynchronous and their errors are *sticky and deferred*, so without an explicit check the failure surfaces at some unrelated later CUDA call with a misleading stack. The Python side is three lines:

```python
# nanoserve/kernels/__init__.py
from torch.utils.cpp_extension import load

_mod = load(
    name="nano_prologue",
    sources=["nanoserve/kernels/prologue.cu"],
    extra_cuda_cflags=["-O3", "-lineinfo"],   # -lineinfo: keep source lines for the debugger
    verbose=True,
)
rmsnorm = _mod.rmsnorm
```

Note `-lineinfo` in the compile flags. It costs nothing at runtime and it is what lets `cuda-gdb` and Compute Sanitizer map a fault back to a line of your source instead of an inscrutable SASS address. Set it now; you will need it in section 7. The first `load()` call blocks for ten to sixty seconds while `nvcc` compiles; after that it is cached in `~/.cache/torch_extensions` keyed on the source hash.

## 5. RoPE: a rotation, and the layout that quietly breaks everything

Rotary position embedding rotates each 2D pair of a head's query and key vectors by an angle proportional to the token's position. The [forward-pass post](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) derives why that makes attention scores depend only on the *relative* offset $m - n$; here we only need the mechanics and the one gotcha that produces wrong logits with no error message.

The cos/sin tables are precomputed once at load time — they are functions of position and frequency only, so recomputing 32 transcendentals per layer per token would be pure waste. The reference builds a `[max_pos, D]` table where the frequencies are concatenated as `(freqs, freqs)`, which means `cos[i] == cos[i + D/2]` and `sin[i] == sin[i + D/2]`. That symmetry is what makes the kernel clean. The pure-PyTorch reference we match uses the **half-split** pairing (`rotate_half`), which is what Hugging Face checkpoints expect:

```python
# nanoserve/model.py — the RoPE reference (half-split / rotate_half)
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def apply_rope(q, k, cos, sin):        # q,k: [..., D]; cos,sin: [..., D]
    q_out = (q.float() * cos + rotate_half(q.float()) * sin).to(q.dtype)
    k_out = (k.float() * cos + rotate_half(k.float()) * sin).to(k.dtype)
    return q_out, k_out
```

Expand `rotate_half` for a single pair. For index $i$ in the first half of a head ($0 \le i \lt D/2$), writing $c = \cos_i$ and $s = \sin_i$:

$$
y_i = x_i \, c - x_{i + D/2} \, s, \qquad y_{i + D/2} = x_{i + D/2} \, c + x_i \, s
$$

That is one 2D rotation of the pair $(x_i,\, x_{i+D/2})$. So the natural kernel assigns **one thread to one pair**: thread $i$ reads $x_i$ and $x_{i+D/2}$, reads $c$ and $s$ from the table, and writes both outputs. With $D/2$ threads per head, thread indexing is trivial and the two reads within each half are contiguous across threads — coalesced.

```cuda
// One block per (token, head). blockDim.x == D/2. Half-split (HF) convention.
template <typename scalar_t>
__global__ void rope_kernel(
    scalar_t* __restrict__ q,            // [n_tokens, n_qh, D] (in place)
    scalar_t* __restrict__ k,            // [n_tokens, n_kh, D]
    const float* __restrict__ cos,       // [max_pos, D]
    const float* __restrict__ sin,
    const int* __restrict__ positions,   // [n_tokens]
    const int n_qh, const int n_kh, const int D) {
  const int tok  = blockIdx.x;
  const int head = blockIdx.y;           // grid.y covers max(n_qh, n_kh)
  const int i    = threadIdx.x;          // 0 .. D/2-1  (one pair per thread)
  const int half = D >> 1;
  if (i >= half) return;

  const int pos = positions[tok];
  const float c = cos[(size_t)pos * D + i];   // cos[i] == cos[i+half]
  const float s = sin[(size_t)pos * D + i];

  if (head < n_qh) {                     // rotate this query head
    scalar_t* base = q + (((size_t)tok * n_qh) + head) * D;
    float x1 = static_cast<float>(base[i]);
    float x2 = static_cast<float>(base[i + half]);
    base[i]        = static_cast<scalar_t>(x1 * c - x2 * s);
    base[i + half] = static_cast<scalar_t>(x2 * c + x1 * s);
  }
  if (head < n_kh) {                     // rotate this key head (GQA: fewer)
    scalar_t* base = k + (((size_t)tok * n_kh) + head) * D;
    float x1 = static_cast<float>(base[i]);
    float x2 = static_cast<float>(base[i + half]);
    base[i]        = static_cast<scalar_t>(x1 * c - x2 * s);
    base[i + half] = static_cast<scalar_t>(x2 * c + x1 * s);
  }
}
```

The fp32 promotion around the rotation is deliberate and matches the reference: a rotation preserves vector norm in exact arithmetic, but in bf16 it does not, and that error compounds over 32 layers. Doing the multiply-add in fp32 and casting back costs nothing on a memory-bound op and keeps the parity check green. This is the same tolerance story as the [sampling-numerics post](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance): the arithmetic order is part of the contract, not an implementation detail.

### The layout gotcha, in kernel form

![Decision tree from the checkpoint source to the rotary pairing convention showing that Hugging Face weights only decode correctly under half split and interleaved silently corrupts them](/imgs/blogs/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope-4.webp)

Here is where hand-written RoPE kernels die. There are two pairing conventions, and they are genuinely different functions:

- **Half-split**: pair $(x_0, x_{D/2})$, $(x_1, x_{D/2+1})$, … — first half with second half. This is what `transformers` and the kernel above implement, and what Hugging Face `.safetensors` checkpoints expect, because those checkpoints ship q/k projection weights that have been *permuted* at conversion time to make half-split reproduce Meta's original result.
- **Interleaved**: pair $(x_0, x_1)$, $(x_2, x_3)$, … — adjacent dimensions. This is Meta's original reference, using complex arithmetic on the un-permuted weights.

Feed a Hugging Face checkpoint through an interleaved kernel and nothing crashes, no NaN appears, and generation produces fluent, plausible, positionally-scrambled garbage. Your kernel indexes `base[i]` and `base[i + half]`; an interleaved kernel would index `base[2*i]` and `base[2*i + 1]`. One character of difference, silent for the rest of the run. The mechanical rule: Hugging Face safetensors → half-split; Meta `consolidated.*.pth` → interleaved; GGUF → check which converter produced it. And the diagnostic from the forward-pass post transfers directly — set `cos = 1, sin = 0` in both your kernel and the reference to make RoPE the identity; if they now agree and previously did not, the bug is in your rotation, in two lines.

#### Worked example: coalescing and vectorized loads for RoPE

Take Llama-3.1-8B: $D = 128$, so `half = 64` and each head gets a 64-thread block (two full warps). Within a warp, the 32 threads read `base[0..31]` (contiguous, one 64-byte segment for bf16) and `base[64..95]` (a second contiguous segment). Two coalesced transactions per warp per read — optimal. If you had instead written the interleaved version with `base[2*i]`, the 32 threads would read 32 addresses two elements apart, doubling the transactions and halving effective bandwidth for no benefit. For an even fatter load, a bf16 kernel can use `__half2` (or a `float4` for a 128-bit transaction) so each thread pulls two adjacent elements at once, cutting the instruction count in half; that is worth doing on wider heads (`head_dim = 256` models) where the pair count is large. The traffic itself is small — per token, RoPE reads and writes the 32 query heads and 8 key heads: $(32 + 8) \cdot 128 \cdot 2 \cdot 2 = 40$ KB per layer, about 1.3 MB across 32 layers. Source: derived from the Llama-3.1-8B config.

## 6. The fusion that is actually possible

Now the payoff. Look again at Figure 1: the pre-attention RMSNorm feeds the QKV projection, and only *after* that projection do we get the q/k tensors that RoPE rotates. A GEMM sits between the norm and the rope. You cannot fuse across it with an elementwise kernel — fusing a matmul with its epilogue is a different, harder kernel (and a later post in this track). So the tempting "fuse RMSNorm + RoPE" is a mirage for a Llama-style model.

But there is a real fusion, and it is exactly what production engines ship. Models like **Qwen3-8B** (in this series' model matrix) apply a per-head RMSNorm to the query and key vectors *after* projection and *before* RoPE — "QK-Norm," a 128-wide normalization of each head. That norm, the rotation, and the KV-cache write are all elementwise on the same projected q/k tensors, all on the far side of the GEMM. Fuse those three and you read q/k once and write once, instead of three separate read-modify-write passes. This is precisely the pattern the vLLM team describes as **HpcRopeNorm** in their [HPC-Ops backend post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06, cited — past the assistant's cutoff): a fused prologue combining "QK-Norm + RoPE + KV-cache write" in one kernel, feeding their Hunyuan attention backend. We are building the toy version of a real thing.

![Before and after comparison contrasting three separate read and write passes for norm rotation and cache write against a single fused pass that reads and writes once](/imgs/blogs/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope-5.webp)

The derivation of the win is pure traffic accounting. For a memory-bound op, runtime is proportional to bytes moved through HBM, so count the passes over the q/k data. Let $B$ be the bytes of the q and k tensors for the batch.

- **Unfused**: QK-Norm reads $B$, writes $B$. RoPE reads $B$, writes $B$. The cache write reads $B$ (from the RoPE output) and writes $B$ (into the cache). Total: $6B$ of HBM traffic.
- **Fused**: read $B$ once, do the norm, rotation, and cache-write in registers, write $B$ once (the K/V goes to the cache, Q goes back to its buffer). Total: $2B$.

So the fused kernel moves one-third of the HBM traffic, and for a bandwidth-bound op the ideal speedup is the traffic ratio, $6B / 2B = 3\times$. Reality lands below the ceiling — cos/sin table reads, block-table lookups for the paged cache, and the fact that L2 sometimes catches the intermediate writes all shave it down — but the *direction* and the *ceiling* are derived, not measured, and that is the honest way to state it. Here is the fused kernel for the QK-Norm case, one block per `(token, head)`:

```cuda
// Fused QK-Norm + RoPE, one block per (token, head), blockDim.x == D.
// Reads the head once into registers, normalizes, rotates, writes once.
template <typename scalar_t>
__global__ void fused_qknorm_rope_kernel(
    scalar_t* __restrict__ q, const scalar_t* __restrict__ q_norm_w, // [D]
    const float* __restrict__ cos, const float* __restrict__ sin,
    const int* __restrict__ positions, const float eps, const int D) {
  const int tok  = blockIdx.x;
  const int head = blockIdx.y;
  const int i    = threadIdx.x;                 // 0 .. D-1
  scalar_t* base = q + (((size_t)tok * gridDim.y) + head) * D;

  // 1) one HBM read, promote to fp32
  float v = (i < D) ? static_cast<float>(base[i]) : 0.0f;

  // 2) RMSNorm over the head dim (reduction is the section-3 pattern)
  float ss = block_reduce_sum(v * v);
  __shared__ float inv_rms;
  if (i == 0) inv_rms = rsqrtf(ss / D + eps);
  __syncthreads();
  float vn = static_cast<float>(static_cast<scalar_t>(v * inv_rms))
             * static_cast<float>(q_norm_w[i]);  // normalized, in registers

  // 3) RoPE, in registers — needs the paired lane's value via shared memory
  __shared__ float buf[256];
  buf[i] = vn; __syncthreads();
  const int half = D >> 1;
  const int pair = (i < half) ? (i + half) : (i - half);
  const int fi   = (i < half) ? i : (i - half);   // table index folds
  const float c = cos[(size_t)positions[tok] * D + fi];
  const float s = sin[(size_t)positions[tok] * D + fi];
  float rotated = (i < half) ? (vn * c - buf[pair] * s)
                             : (vn * c + buf[pair] * s);

  // 4) one HBM write
  base[i] = static_cast<scalar_t>(rotated);
}
```

The interesting complication is step 3: RoPE pairs element $i$ with element $i \pm D/2$, but in this layout each thread owns a *single* element $i$ of the head, not a pair, so a thread needs its partner's *normalized* value. That value lives in another thread's register, so we stage the whole normalized head through shared memory (one write, one barrier) and read the partner. This is a genuine design tension worth naming: the RMSNorm reduction wants one thread per element (to parallelize the sum), but the RoPE rotation wants one thread per pair (to have both operands locally). Staging through shared memory reconciles them at the cost of one `__syncthreads()` — cheaper than a second HBM round-trip by a wide margin. In `nanoserve` I keep both a standalone `rope_kernel` (for Llama, no QK-Norm) and this `fused_qknorm_rope_kernel` (for Qwen3-style models), and the model config picks which one runs.

## 7. Correctness first, always

A CUDA kernel has two failure modes and they need different tools. It can *crash* — an illegal memory access, a launch failure — which is loud and locatable (section 8). Or it can *silently compute the wrong thing*, like the interleaved-RoPE bug, which is quiet and will ship to production if nothing catches it. The parity harness catches the second kind, and it is the most important 30 lines in this post. It reuses the pure-PyTorch reference from the forward-pass post as ground truth — the same discipline as that post's `verify.py`, extended to kernels.

![Dataflow of the parity harness where one fixed input feeds both the CUDA kernel and the PyTorch reference whose outputs meet at a tolerance check that branches to ship or bisect](/imgs/blogs/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope-6.webp)

```python
# nanoserve/tests/test_prologue.py
import torch
from nanoserve.kernels import rmsnorm            # our CUDA op
from nanoserve.model import rms_norm as rms_ref  # pure-PyTorch reference

def test_rmsnorm_matches_reference():
    torch.manual_seed(0)                          # fixed input, reproducible
    S, H, eps = 128, 4096, 1e-5
    x = torch.randn(S, H, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

    out_cuda = rmsnorm(x, w, eps)
    out_ref  = rms_ref(x, w, eps)

    # bf16 tolerance, NOT 1e-8. See the sampling-numerics post: two correct
    # implementations differ by units-in-the-last-place; demand fp32 exactness
    # of a bf16 op and every correct kernel "fails".
    torch.testing.assert_close(out_cuda, out_ref, rtol=1.6e-2, atol=1e-2)
    # A stricter, honest cross-check: run BOTH in fp32 and tighten the bound.
    out_cuda32 = rmsnorm(x.float(), w.float(), eps)
    out_ref32  = rms_ref(x.float(), w.float(), eps)
    torch.testing.assert_close(out_cuda32, out_ref32, rtol=1e-5, atol=1e-6)
```

Two tolerances, on purpose. The bf16 comparison uses `rtol` and `atol` around $10^{-2}$ because — as the [sampling-numerics post](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance) argues at length — two *correct* bf16 implementations that accumulate in a different order disagree in the last few bits, and demanding `1e-8` makes correctness impossible to prove. The fp32 comparison then tightens the bound to catch real logic bugs that the loose bf16 tolerance would hide: if your kernel is off by a genuine factor and not just rounding, the fp32 test flags it while the bf16 test might not.

**When it fails, bisect the compose.** The fused kernel does three things; a mismatch could be in any of them. Do not stare at the fused kernel. Test the pieces in isolation, in order:

```python
# Bisect a failing fused kernel by testing each stage against the reference.
def bisect_fused(q, qnw, cos, sin, pos, eps):
    # 1) QK-Norm alone: run the fused kernel with cos=1, sin=0 (RoPE = identity)
    ones, zeros = torch.ones_like(cos), torch.zeros_like(sin)
    got  = fused_qknorm_rope(q.clone(), qnw, ones, zeros, pos, eps)
    want = rms_ref_per_head(q, qnw, eps)              # norm only
    torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-2)  # norm OK?

    # 2) RoPE alone: set q_norm_w = 1 and feed pre-normalized q
    unit = torch.ones_like(qnw)
    got  = fused_qknorm_rope(q.clone(), unit, cos, sin, pos, eps=0.0)
    want = apply_rope_ref(rms_ref_per_head(q, unit, 0.0), cos, sin, pos)
    torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-2)   # rope OK?
    # If (1) passes and (2) fails, your bug is in the RoPE stage — start there.
```

Setting `cos = 1, sin = 0` turns RoPE into the identity, so stage 1 isolates the norm; setting `q_norm_w = 1` and `eps = 0` neutralizes the norm's scale so stage 2 isolates the rotation. Nine times out of ten the failure is one of three things: the RoPE layout (half-split vs interleaved), the epsilon placement, or the cast-order in the weight multiply — all three visible in isolation, invisible in the fused whole.

## 8. Debugging the segfault (your first kernel will have one)

Your first CUDA kernel indexes past the end of an array. Everyone's does. The symptom is an "illegal memory access" that, because launches are asynchronous, surfaces at some later, innocent-looking CUDA call. The blunt first move is `CUDA_LAUNCH_BLOCKING=1`, which serializes launches so the error at least points at the right kernel — but it cannot locate the failing *thread*, and inside a CUDA graph it only shows the graph launch point. The vLLM team documents the better workflow in ["Debugging CUDA memory-access bugs with a GPU core dump"](https://vllm.ai/blog/2025-08-11-cuda-debugging) (2025-08-11, cited): let the GPU dump core on the illegal access and open it in the debugger.

```bash
# Enable a GPU core dump on the illegal access (per the vLLM CUDA-debugging post)
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
# Shrink the dump so it is openable; do NOT add skip_abort (documented bug)
export CUDA_COREDUMP_GENERATION_FLAGS="skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory"

python -m nanoserve.tests.test_prologue     # crashes, writes the core file

cuda-gdb                                     # then, in the debugger:
#   (cuda-gdb) target cudacore /tmp/cuda_coredump_host.12345.167...
#   -> reports the failing kernel, grid/block/thread, SM/warp/lane, and
#      CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS (14) at your source line
```

For the debugger to name a *line* of your `.cu` instead of a raw address, the extension must be built with line info — which is why section 4 put `-lineinfo` in `extra_cuda_cflags`. If you skipped it, rebuild first: `NVCC_PREPEND_FLAGS='-lineinfo'` in the environment forces it globally without editing the build. The vLLM post is candid that even with line info the debugger sometimes fails to resolve the exact line and shows only the last line after inlining — so this is a strong first tool, not a magic wand.

Once the core dump names the neighborhood, **Compute Sanitizer** confirms the class of bug. It is the CUDA equivalent of Valgrind and it catches the three bugs every first kernel has:

```bash
compute-sanitizer --tool memcheck  python -m nanoserve.tests.test_prologue  # OOB reads/writes
compute-sanitizer --tool racecheck python -m nanoserve.tests.test_prologue  # shared-mem races
compute-sanitizer --tool synccheck python -m nanoserve.tests.test_prologue  # divergent __syncthreads
```

![Matrix mapping four first kernel symptoms to their root cause the tool that finds it and the one line fix](/imgs/blogs/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope-7.webp)

The three bugs, in the order you will meet them:

- **Out-of-bounds thread index.** You launched 256 threads but the last iteration of a grid-stride loop, or a missing `if (i >= half) return;`, reads element `H + something`. `memcheck` points at the exact load. The fix is the guard you forgot.
- **Missing `__syncthreads()`.** A thread reads `warp_sums[]` or the staged RoPE buffer before every warp has written its slot, so it reads stale or uninitialized data. This one is *insidious* because it often produces the right answer under light load and the wrong answer when the scheduler happens to interleave differently — a nondeterministic bug. `synccheck` and a careful read of your barriers find it.
- **Race on shared memory.** Two threads write the same shared slot, or one reads while another writes, with no barrier ordering them. `racecheck` names the conflicting accesses. In the reductions above, the single `__syncthreads()` between the warp-sum writes and the first-warp read is the barrier that prevents this; delete it and `racecheck` lights up.

There is a second, subtler failure the vLLM team's [follow-up post](https://vllm.ai/blog/2025-12-03-improved-cuda-debugging) (2025-12-03, cited) addresses: a kernel that *hangs* rather than crashes — usually a `__syncthreads()` that only some threads of a block reach, so the barrier never completes. For that you trigger a core dump on demand (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1` plus a pipe) and inspect the stuck kernel. If you ever write a barrier inside an `if` that not all threads take, you have written this bug.

## 9. The numbers, with provenance

Both kernels are memory-bound, and the way to prove it is the roofline — the model from the [HPC roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound). **Arithmetic intensity** is FLOPs per byte of HBM traffic, $\text{AI} = \text{FLOPs} / \text{bytes}$. RMSNorm does about five floating-point operations per element (square, add, the reciprocal-sqrt amortized, multiply, scale) and moves 4 bytes per element (a bf16 read plus a bf16 write). So its intensity is roughly one FLOP per byte. RoPE is even leaner: about six FLOPs per pair against 8 bytes moved, well under one FLOP per byte.

Compare that to the ridge point — the intensity where a GPU stops being memory-bound and becomes compute-bound. An A100's bf16 tensor-core throughput is about 312 TFLOP/s against ~2.0 TB/s of HBM (NVIDIA A100 datasheet), a ridge near 150 FLOP/byte; even on plain fp32 CUDA cores the ridge is around 10 FLOP/byte. Our kernels sit at ~1, a hundred-fold to the left of the ridge. They are memory-bound by two orders of magnitude, which is *why* the entire optimization story is "move fewer bytes" (fusion, vectorized loads) and never "do less math."

For a memory-bound op the runtime floor is simply bytes divided by bandwidth. The table derives that floor for the per-token, per-layer traffic of each kernel on the series' hardware matrix.

| Kernel (per token, per layer) | Bytes moved | Time @ 4090 ~1.0 TB/s | Time @ A100 ~2.0 TB/s | Time @ H100 ~3.35 TB/s | Source |
| --- | --- | --- | --- | --- | --- |
| RMSNorm (H=4096, bf16) | 16 KB | ~16 ns | ~8 ns | ~5 ns | derived |
| RoPE (40 heads × 128, bf16) | 40 KB | ~40 ns | ~20 ns | ~12 ns | derived |
| Unfused QK-Norm+RoPE+write | ~120 KB | ~120 ns | ~60 ns | ~36 ns | derived |
| Fused prologue (this post) | ~40 KB | ~40 ns | ~20 ns | ~12 ns | derived |
| 4090 HBM bandwidth | — | 1008 GB/s | — | — | cited: NVIDIA RTX 4090 spec |
| A100 80GB SXM bandwidth | — | — | 2039 GB/s | — | cited: NVIDIA A100 datasheet |
| H100 SXM bandwidth | — | — | — | 3350 GB/s | cited: NVIDIA H100 datasheet |

Two honest caveats on that table. First, these are *bandwidth floors*, not predicted runtimes — a real kernel adds launch overhead (a few microseconds, which for these nanosecond-scale ops dominates entirely at batch 1) and never hits 100% of peak bandwidth (70–85% achieved is typical). Second, the fused row equals the RoPE row because fusion collapses the three passes into the one-read-one-write of a single elementwise sweep; the $3\times$ is the ratio of the unfused row to the fused row, derived in section 6.

### How to measure it honestly

If you want a real number, the [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) has the full recipe; the CUDA-specific essentials are:

```python
# nanoserve/bench/prologue_bench.py — the honest way to time a kernel
import torch
def time_kernel(fn, *args, warmup=25, iters=100):
    for _ in range(warmup):              # warm caches, autotune, JIT
        fn(*args)
    torch.cuda.synchronize()             # the CPU is ahead of the GPU; wait
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()             # kernels are async; sync before read
    return start.elapsed_time(end) / iters   # milliseconds per call
```

The two lines beginners omit are both `torch.cuda.synchronize()` — because CUDA kernels are asynchronous, the CPU races ahead and any timing that does not sync before reading the clock measures launch latency, not execution. Use CUDA events, not `time.time()`, so you time GPU work rather than Python. Warm up first (the JIT compile, the caches, any autotuning happen once). And for a headline number, lock the clocks (`nvidia-smi -lgc <freq>`) so boost-clock variance does not turn your measurement into noise. Then compute achieved bandwidth as bytes-moved divided by measured-time and compare it to the datasheet peak — for a memory-bound kernel, *fraction of peak bandwidth* is the honest scoreboard, not tok/s.

#### Worked example: the fused prologue on a 4090, framed as reproduce-it-yourself

On an RTX 4090 (~1.0 TB/s), the fused prologue for one Qwen3-8B token-layer moves ~40 KB, a bandwidth floor near 40 ns. You will not see 40 ns per call at batch 1 — you will see the launch overhead, a few microseconds, because the op is far too small to hide it. That is the real lesson: at batch 1 these kernels are *launch-bound*, not bandwidth-bound, which is the entire argument for CUDA graphs (capture the launches once, replay with near-zero overhead) and for fusion (one launch instead of three). Run `prologue_bench.py` at batch 1 and again at batch 256; you should see per-token time fall sharply as the fixed launch cost amortizes across the batch and the kernel finally becomes bandwidth-bound. Report the batch where the crossover happens on your card. Source: reproduce with `prologue_bench.py`; the bandwidth floor is derived, the launch-overhead claim is the standard reason CUDA graphs exist.

## 10. Stress tests and edge cases

A kernel that works on `S=128, H=4096` and nothing else is a demo, not an engine. Four stresses separate the two.

**Hidden size not divisible by the block or warp size.** If `H = 4096` and `blockDim = 256`, the grid-stride loop covers it exactly. But `head_dim` is not always a clean multiple of 32 — and more to the point, the fused kernel launches `blockDim.x == D` threads, so a head dim that is not a multiple of the warp size leaves a partial final warp. `block_reduce_sum` already handles this: the guard `val = (threadIdx.x < n_warps) ? warp_sums[lane] : 0.0f` zero-fills the missing lanes, and the grid-stride loop naturally covers a ragged tail. The bug to avoid is assuming `H % blockDim.x == 0` and dropping the tail elements — always write the loop as `for (i = tid; i < H; i += stride)`, never `for (i = tid*chunk; i < (tid+1)*chunk; i++)`.

**bf16 vs fp32 accumulation drift.** Swap the fp32 accumulator in the reduction for a bf16 one and the parity check fails — not because the kernel is "wrong" but because summing thousands of bf16 squares in bf16 loses precision the reference does not lose. This is the exact batch-invariance concern the vLLM team raises in their [bitwise-consistent determinism post](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) (2025-11-10, cited): they make RMSNorm batch-invariant precisely because reduction order and precision change the result. Keep the accumulator fp32; the cost is nil on a memory-bound op.

**A ragged batch of flattened sequences.** [Continuous batching](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) does not pad — it flattens all sequences into one long `[total_tokens, H]` tensor with a `positions` array giving each token its RoPE angle. The RMSNorm kernel does not care: one block per row, and a row is a row. The RoPE kernel does care, because position is per-token — which is exactly why the kernel above takes a `positions[]` array indexed by `blockIdx.x` rather than assuming `pos == blockIdx.x`. Pass the real positions and the same kernel serves prefill, decode, and a mixed continuous batch without change. That one design choice — positions as data, not derived from the loop index — is what makes the kernel reusable across the whole engine.

**When *not* to write your own.** Be honest with yourself before you ship a hand-CUDA kernel: `torch.compile` will often fuse RMSNorm, the residual add, and elementwise ops into a single Inductor-generated kernel automatically, and the vLLM team reports fusions like SiLU+Quant and AllReduce+RMSNorm buying single-digit to ~15% end-to-end in their [torch.compile post](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20, cited). If Inductor already generates a kernel as fast as yours, your hand-written version is pure maintenance liability. Write your own only when you need a fusion the compiler will not find — QK-Norm + RoPE + a *paged* KV-cache write across a block table is exactly such a case, because the block-table indirection is opaque to Inductor — or a dtype/layout no library supports. Measure against `torch.compile` and PyTorch's native ops first; the next post in this track ports these same kernels to Triton, and the [when-to-stop-writing-CUDA argument](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) is real.

## 11. Case studies and public numbers

Four public results anchor the claims above, each cited with its setup so you can check it.

**HpcRopeNorm, the production version of this post's fusion.** The vLLM team's [HPC-Ops backend post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06, cited) describes a fused prologue "HpcRopeNorm = QK-Norm + RoPE + KV-cache write" feeding a three-stage persistent attention kernel for Tencent's Hunyuan models. They report the *attention* backend reaching up to 2.95× over a static split-KV baseline and about 2.25× over FlashInfer/FlashAttention on an H20 with Hunyuan-3 in FP8 — the RopeNorm fusion is one component of that stack, not separately benchmarked, so cite it as the pattern, not as a standalone speedup.

**Warp-shuffle reductions, the canonical source.** NVIDIA's ["Faster Parallel Reductions on Kepler"](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) is the reference implementation of the `__shfl_down` block reduction used here; it is where the shared-memory-free warp reduction became standard practice, and it predates the tensor-core era by a decade — the pattern is that durable.

**GPU core dumps for illegal-access bugs.** The vLLM CUDA-debugging posts ([2025-08-11](https://vllm.ai/blog/2025-08-11-cuda-debugging) and [2025-12-03](https://vllm.ai/blog/2025-12-03-improved-cuda-debugging), both cited) document the exact `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` / `cuda-gdb target cudacore` workflow used in section 8, including the honest limitations (line resolution often fails after inlining; a running kernel cannot be interrupted with Ctrl-C).

**Reduction precision as a correctness issue, not a perf one.** The vLLM [bitwise-determinism post](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) (2025-11-10, cited) makes RMSNorm and the SiLU MLP batch-invariant to get bitwise-identical outputs across batch sizes, and reports that full determinism costs about 2.4× on an RL run — direct evidence that the fp32-accumulation and reduction-order decisions in this post's kernels are load-bearing, not pedantry.

## When to reach for this (and when not to)

Write these kernels by hand when you are *learning* — there is no substitute for having written a block reduction and a coalesced load with your own fingers, and every later kernel in this track assumes you have. Write them for production when you need a fusion or a layout the compiler cannot express: the QK-Norm + RoPE + paged-KV-write prologue is the real example, because the block-table indirection defeats `torch.compile`, and because collapsing three launches into one matters at the batch sizes where you are launch-bound.

Do *not* write them for production when PyTorch, `torch.compile`, or an existing library already does the job. A standalone RMSNorm is a solved problem — Apex, FlashInfer, and vLLM ship tuned ones, and Inductor generates a competitive one for free. A hand kernel that merely ties the library is negative value: it is code you now own, must test on every new GPU, and must keep matching the reference as dtypes evolve. The decision rule is a single measurement: if your kernel is not meaningfully faster than `torch.compile`'s output on your target hardware and batch, delete it and use the compiler. Reach for CUDA when the profiler shows a fusion opportunity the compiler is leaving on the table — and prove it with the honest bandwidth measurement, not a vibe.

## Key takeaways

- **A kernel is one function every thread runs with a different index.** For RMSNorm, one block owns one token row; different tokens are independent blocks. That single mapping decision drives the whole design.
- **Reduce in fp32, always.** bf16 accumulation of thousands of squares loses precision the reference does not, and your parity check will fail for a reason that looks like a bug but is a numerics choice.
- **The warp-shuffle reduction beats naive shared memory** by removing barriers, avoiding bank conflicts, and staying in registers — five `__shfl_down_sync` steps collapse 32 lanes to one.
- **Coalesce your loads.** Adjacent threads must read adjacent memory; the grid-stride loop gives you this for free, and the chunked split silently throws away most of your bandwidth.
- **RoPE's layout is the silent killer.** Half-split for Hugging Face checkpoints, interleaved for Meta's; the wrong one produces fluent, positionally-scrambled output with no error. Zero out cos/sin to isolate it.
- **The fusable prologue is QK-Norm + RoPE + KV-write**, not RMSNorm + RoPE — a GEMM sits between the pre-attention norm and the rotation. Fusing the three post-projection elementwise ops moves one-third the HBM traffic.
- **Prove correctness against the pure-PyTorch reference** with a bf16 tolerance around $10^{-2}$ and a tighter fp32 cross-check; bisect a failing fused kernel by neutralizing one stage at a time.
- **Your first kernel will segfault.** Build with `-lineinfo`, dump GPU core on the illegal access, open it in `cuda-gdb`, and confirm the bug class with Compute Sanitizer.
- **These ops are memory-bound at ~1 FLOP/byte**, a hundredfold left of the roofline ridge; the only optimization that helps is moving fewer bytes, and the honest scoreboard is fraction-of-peak-bandwidth, not tok/s.
- **Reach for hand-CUDA only when the compiler leaves a fusion on the table** — otherwise `torch.compile` and library kernels win on both speed and maintenance.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series map and the honesty rule these numbers obey.
- [A forward pass by hand: Llama from scratch](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) — the pure-PyTorch RMSNorm and RoPE reference this post matches against.
- [The inference kernel landscape](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs) — the ~10 kernels of one decode step and where these two sit.
- [Sampling numerics, determinism, and batch invariance](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance) — why fp32 accumulation and reduction order are correctness, not taste.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone decision guide, including when to stop writing CUDA.
- [CUDA programming for AI engineers: threads, blocks, and a first kernel](/blog/machine-learning/high-performance-computing/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel) — the from-zero CUDA foundation this post assumes.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the arithmetic-intensity argument in full.
- [NVIDIA's "Faster Parallel Reductions on Kepler"](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) — the canonical warp-shuffle reduction, and the [RoPE paper](https://arxiv.org/abs/2104.09864) and [RMSNorm paper](https://arxiv.org/abs/1910.07467) for the math.
