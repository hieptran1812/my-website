---
title: "Kernel Fusion, CUDA Graphs, and torch.compile: Cutting Per-Step Overhead in LLM Decode"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Why LLM decode is launch-bound at small batch, and how torch.compile, operator fusion, and CUDA graphs turn thousands of tiny kernel launches per token into a single launchless replay."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "torch-compile",
    "cuda-graphs",
    "kernel-fusion",
    "gpu-optimization",
    "llm-inference",
    "latency",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-1.webp"
---

The first time I profiled a "slow" Llama-8B deployment, the GPU was the innocent party. The A100 sat at 30% utilization while the service missed its per-token latency target by 2x. No OOM, no throttling, no thermal wall. The GPU was *starving*. It kept running out of work and waiting for the CPU to hand it the next kernel — and the CPU could not enqueue kernels fast enough, because a single decode step for that model fires well over a thousand tiny GPU kernels, each of which costs the CPU several microseconds just to launch. At batch size 1, the model was not compute-bound. It was launch-bound.

That is the failure mode this post is about, and it is one of the most misunderstood in production LLM serving. Every technique here — `torch.compile`, operator fusion, and CUDA graphs — is a different way to attack the *same* enemy: per-step overhead that has nothing to do with how fast your GPU can do math. It is the CPU-side cost of telling the GPU what to do, one op at a time. When decode is launch-bound, the answer is not a bigger GPU or a lower-precision matmul; it is to stop launching so many kernels. The figure below is the whole thesis in one picture.

The reason this matters *now*, more than it did five years ago, is a widening gap. GPU compute and memory bandwidth have grown fast — an H100 does math several times faster than a V100 — but the CPU-side cost of launching a kernel has barely moved, because it is set by the operating system, the CUDA driver, the PyTorch dispatcher, and the Python interpreter, none of which got dramatically faster. So the *fraction* of a small-batch decode step spent on launch overhead has grown with every hardware generation. Techniques that were nice-to-haves on older GPUs are now the difference between 30% and 90% utilization on a modern one. The three levers below — a compiler front end that removes Python dispatch, a fusion pass that removes both kernels and memory traffic, and a graph-capture mechanism that removes per-launch cost entirely — are how you close that gap, and they are why every serious LLM serving engine ships them on by default.

![Side by side comparison of eager decode firing many kernel launches versus a CUDA graph replaying the whole step in one launch](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-1.webp)

By the end of this post you will be able to: (1) *prove* whether a given deployment is launch-bound with a two-line calculation and a profiler trace; (2) apply `torch.compile(mode="reduce-overhead")` to a serving model and understand exactly what it does to the dispatch path; (3) capture and replay a CUDA graph by hand with `torch.cuda.CUDAGraph`, and know why serving stacks pad requests to bucketed batch sizes to make graphs reusable; (4) read `TORCH_LOGS="recompiles"` to diagnose a recompilation storm; and (5) measure token-per-output-time (TPOT) before and after on named hardware across batch sizes, so you can say precisely when this work is worth doing and when it is a waste of engineering time.

This is the compiler-and-runtime path to lower latency. It is the sibling of the hand-written-kernels path — where you drop into CUDA or Triton and fuse operations yourself. Both attack overhead; this one lets the compiler and the CUDA driver do the fusing and the launch-batching for you. If you want the hand-authored version, read [Custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference). Everything here sits on the same SLO triangle as the rest of the series: latency, throughput, and cost are the three corners, and per-step overhead is a tax you pay on all three at once when you get it wrong.

## The anatomy of a launch-bound decode step

Start with the shape of the work. An autoregressive decode step produces exactly *one* token. To do that it runs a full forward pass through the model: for a dense transformer with $L$ layers, each layer executes an attention block (a QKV projection, rotary position embedding, the attention itself, an output projection) and an MLP block (a gate projection, an up projection, an activation, an element-wise multiply, a down projection), plus two normalization layers and two residual adds. In eager PyTorch, almost every one of those is a separate kernel launch — and several of them decompose into multiple kernels. Count it up and an 8B-parameter model lands somewhere around 1,000 to 2,000 kernel launches to produce a single token.

Here is the part that trips people up. During *prefill* — processing the prompt — each of those kernels operates on hundreds or thousands of tokens at once, so each kernel has real work to do and runs for a meaningful chunk of time. During *decode*, the same kernels operate on a single new token per sequence. A matmul that was a fat GEMM in prefill becomes a skinny GEMV in decode: a matrix times a vector. That operation is memory-bound and finishes in a couple of microseconds. The GPU blinks and it is done.

Now the arithmetic that decides everything. Let:

- $N$ = number of kernel launches per decode step (roughly 1,000–2,000 for an 8B model),
- $\tau$ = the CPU-side cost to prepare and enqueue *one* kernel — Python interpreter overhead, the PyTorch dispatcher, autograd bookkeeping, and the `cudaLaunchKernel` driver call — which in eager mode runs about 5–30 µs per op depending on the op and the framework version,
- $g(b)$ = the GPU execution time of one kernel at batch size $b$.

CUDA is asynchronous: the CPU enqueues kernel $k$ and immediately moves on to enqueue kernel $k{+}1$ while the GPU chews on $k$. So the two timelines overlap. The step time is governed by whichever side is slower:

$$\text{TPOT} \approx N \cdot \max\big(\tau,\ g(b)\big).$$

That single `max` is the whole story. Two regimes fall out of it:

- **Launch-bound** ($g(b) < \tau$): the GPU finishes each kernel faster than the CPU can queue the next one. The GPU drains the queue and idles in a *bubble* between launches. Step time collapses to $\text{TPOT} \approx N\tau$ — set entirely by the CPU, independent of how fast the GPU is.
- **Compute-bound** ($g(b) > \tau$): each kernel runs long enough that the CPU's launch overhead is fully hidden behind GPU work. Step time is $\text{TPOT} \approx N\cdot g(b)$ — the GPU is the bottleneck, which is what you want.

Batch size is the dial between them. Increasing $b$ increases $g(b)$ (more rows in the GEMV, more work per kernel) while leaving $\tau$ essentially constant (queuing one kernel costs the same whether it processes 1 token or 64). So there is a crossover batch $b^\*$ where $g(b^\*) = \tau$. Below it you are launch-bound; above it you are compute-bound. The figure traces the two timelines op by op.

![Grid timeline showing CPU launch gaps and GPU idle bubbles in eager mode versus a dense back-to-back graph replay](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-2.webp)

The top row is the CPU enqueuing kernels every ~5 µs. The middle row is the GPU: it runs a 2 µs kernel, then sits idle waiting for the next launch to arrive. Those white-and-red gaps are wasted GPU-seconds you are paying for. The bottom row is what a captured graph does: one launch call kicks off the entire recorded sequence, and the kernels run back-to-back with no bubbles.

### Where the thousand kernels come from

It is worth counting the kernels explicitly, because the number surprises people and because it tells you which ops fusion and graphs will actually help. Walk one transformer layer in eager PyTorch, Llama-style:

- **Input RMSNorm** — the mean-of-squares is a reduction, the reciprocal-sqrt and the scale are pointwise; 2–4 kernels depending on how it decomposes.
- **QKV projection** — one matmul kernel (or three if Q, K, V are separate linears).
- **RoPE** — the rotary embedding is a gather plus a couple of pointwise ops; 2–3 kernels.
- **Attention** — if you use a fused kernel (FlashAttention, a fused SDPA backend), this is 1 kernel. Unfused, it is the QKᵀ matmul, a scale, a softmax that decomposes into max/exp/sum/divide, and the ×V matmul — easily 6–8 kernels.
- **Output projection** — one matmul.
- **Residual add** — one pointwise kernel.
- **Post-attention RMSNorm** — another 2–4 kernels.
- **MLP** — gate projection (matmul), up projection (matmul), SiLU activation (pointwise), element-wise multiply (pointwise), down projection (matmul); 5 kernels.
- **Residual add** — one pointwise kernel.

That is roughly 20–30 kernels per layer with fused attention, and 30–40 without. A 32-layer model lands at 650–1,300 kernels for the transformer stack alone. Add the token embedding lookup, the final norm, the LM-head matmul, and the sampling path — argmax or top-k/top-p, which in a naive implementation is several kernels plus a `.item()` — and a full decode step comfortably reaches 1,000–2,000 kernel launches. The key observation for optimization: the *matmuls* are a small fraction of the kernel count but most of the FLOPs, while the *pointwise and norm ops* are most of the kernel count and almost none of the FLOPs. Launch overhead is dominated by the cheap ops. That is exactly the population fusion and graph capture are built to collapse.

### Measuring τ and N without guessing

Do not estimate launch-bound status from intuition — measure it. Two tools give you the answer directly.

The PyTorch profiler attributes time to CPU and CUDA and lets you see the launch-versus-compute split:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        decode_step()
    torch.cuda.synchronize()

# Sort by CUDA time to see where GPU work goes; the CPU/CUDA totals reveal
# whether you are launch-bound (CPU busy, GPU idle) or compute-bound.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
prof.export_chrome_trace("decode_trace.json")   # open in chrome://tracing
```

The definitive view is a timeline profiler such as Nsight Systems:

```bash
nsys profile -o decode --trace=cuda,nvtx python serve.py
```

Open the report and look at two rows: the **CUDA API** row (where `cudaLaunchKernel` calls live) and the **CUDA HW** row (where kernels actually execute). The launch-bound signature is unmistakable — the CUDA API row is nearly saturated with back-to-back launch calls while the CUDA HW row is full of gaps between short kernels. If instead the HW row is dense and the API row is sparse, you are compute-bound and this whole post will buy you little. That single visual is worth more than any back-of-envelope estimate, and it is the first thing I pull up when a decode service misses TPOT with the GPU showing low utilization.

#### Worked example: is my deployment launch-bound?

Take Llama-3.1-8B on an A100-80GB, bf16, batch 1. Suppose $N = 1{,}400$ kernels per token and eager per-op overhead $\tau = 8$ µs. Then the CPU-imposed floor is

$$\text{TPOT}_{\text{launch}} \approx N\tau = 1{,}400 \times 8\,\mu s = 11.2\ \text{ms/token},$$

which caps you at about **89 tokens/s no matter how fast the A100 is**. Meanwhile the actual GPU work for one decode step — reading ~16 GB of bf16 weights plus the KV cache from HBM at ~2 TB/s — is roughly $16\text{ GB} / 2\text{ TB/s} \approx 8$ ms of memory-bound compute. Since $g(b{=}1)$-summed ($\approx 8$ ms) is *below* the launch floor (${11.2}$ ms), you are launch-bound: the CPU, not the GPU, is setting your token rate. Killing launch overhead should pull TPOT down toward the 8 ms memory floor — roughly a 1.4x speedup — and on a faster GPU where the memory read is quicker, the relative win is even bigger because the fixed launch floor dominates more.

That last point is why this technique matters *more* on newer hardware. An H100 reads weights faster (~3.35 TB/s), so its GPU work per step shrinks while the CPU launch cost stays put. The faster your accelerator, the more of your decode latency is pure launch overhead, and the more you have to gain by removing it. Buying a better GPU without fixing launch overhead can leave most of that GPU idle. (For the deeper reason decode is memory-bound in the first place — the KV cache and weight-streaming wall — see [Why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different).)

## torch.compile for serving: what the stack actually does

`torch.compile` is the highest-leverage single line you can add to a PyTorch serving model, and it is also the one most likely to blow up in production if you do not understand its failure modes. It is not a black box that "makes things fast." It is a four-stage compilation pipeline, and each stage removes a distinct source of the overhead we just quantified.

![Layered diagram of the torch.compile stack from the Python module down through Dynamo, AOTAutograd, Inductor, Triton, and a CUDA graph](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-3.webp)

**TorchDynamo** is the front end. It hooks into the CPython frame evaluation API and traces your Python bytecode as it runs, extracting an FX graph of tensor operations. Crucially, it inserts *guards*: runtime checks on the properties it assumed while tracing — input dtypes, tensor shapes, Python constants, the identity of the model. If a guard passes on the next call, the compiled artifact is reused; if it fails, Dynamo recompiles. When Dynamo hits something it cannot trace (a data-dependent Python branch, a call into an opaque C extension, a `.item()` that forces a device sync), it inserts a *graph break*: it compiles the traceable region, drops back to the eager interpreter for the untraceable bit, and resumes. Graph breaks are the silent killer of compiled serving performance, because each break re-introduces the Python overhead you were trying to eliminate.

**AOTAutograd** traces through the FX graph, decomposes composite operators into a smaller set of primitive `aten` ops, and (for training) captures the backward pass. For inference you mostly care that it normalizes the graph into something the backend can pattern-match and fuse.

**TorchInductor** is the backend that does the real work. It takes the normalized graph and generates fused kernels — for GPUs, it emits **Triton**. This is where operator fusion happens: Inductor groups compatible pointwise and reduction operations into single kernels so intermediate results never leave the GPU's registers or shared memory (more on this in the next section).

**CUDA graphs** are the final layer, and this is what `mode="reduce-overhead"` turns on. After Inductor has produced fused Triton kernels, the reduce-overhead mode records the entire sequence of launches into a CUDA graph so that subsequent steps replay in a single launch instead of $N$ launches. Fusion cuts $N$; the CUDA graph cuts the per-launch cost of whatever $N$ remains.

Here is the minimal serving invocation:

```python
import torch

model = load_model().eval().cuda().to(torch.bfloat16)

# reduce-overhead = Inductor fusion + CUDA-graph capture on the hot path.
# fullgraph=True makes Dynamo *raise* on a graph break instead of silently
# falling back — you want this in serving so breaks fail loudly in testing.
compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)

# The FIRST call triggers tracing + compilation (tens of seconds). Warm it up
# BEFORE you route traffic, once per shape you will actually serve.
with torch.inference_mode():
    for _ in range(3):
        _ = compiled(warmup_input)   # compile + capture on this shape
torch.cuda.synchronize()
```

Three things about this snippet decide whether it helps or hurts in production:

- **`fullgraph=True` is a serving safety net, not a nicety.** Without it, a graph break degrades you silently to partial eager execution and you will spend a day wondering why the compiled model is only 10% faster. With it, breaks throw during testing so you fix them before they reach traffic.
- **Warmup is not optional and it is not free.** The first call to a compiled model runs the whole compilation pipeline — commonly 30 to 120 seconds for a large model, sometimes more. If a request triggers that, its TTFT is measured in *minutes*. You must warm up every shape you will serve during startup, before the readiness probe passes.
- **`mode` matters.** `mode="default"` fuses but does not capture CUDA graphs. `mode="reduce-overhead"` adds the graph capture and is the right choice for latency-sensitive, small-batch decode. `mode="max-autotune"` additionally autotunes kernel configs and Triton templates, which lengthens compilation substantially in exchange for the fastest steady-state kernels — worth it for a long-lived server, painful for anything that restarts often.

The mechanics of *why* it helps map straight back to the `max(τ, g(b))` model. `torch.compile` in default mode shrinks $\tau$ by removing Python dispatch overhead per op and by reducing $N$ through fusion. `reduce-overhead` mode then collapses the residual launches into a graph replay, driving the CPU cost per step toward a constant that is independent of $N$. The PyTorch 2 paper (Ansel et al., ASPLOS 2024) reports a geometric-mean inference speedup of about 2.27x on an A100 across 180+ models in TorchBench, HuggingFace, and TIMM — and the biggest wins skew toward exactly the small-batch, many-small-kernel regime that decode lives in.

### Choosing a compile mode

The `mode` argument is the coarsest and most important knob, and picking the wrong one either leaves performance on the table or makes startup unbearable. The four options trade compile time for steady-state speed:

| Mode | Fusion | CUDA graphs | Autotune | Compile time | Best for |
|---|---|---|---|---|---|
| `default` | yes | no | no | seconds–tens of s | throughput serving where launches are already hidden |
| `reduce-overhead` | yes | yes | no | tens of s | latency-sensitive small-batch decode (the LLM case) |
| `max-autotune` | yes | yes | yes | minutes | long-lived servers wanting the fastest steady-state kernels |
| `max-autotune-no-cudagraphs` | yes | no | yes | minutes | autotuned kernels where graphs are handled elsewhere (e.g. by the engine) |

For interactive LLM decode, `reduce-overhead` is the default choice: you want the CUDA-graph replay because you are launch-bound, and you usually cannot afford the multi-minute `max-autotune` compile on every deploy. Reserve `max-autotune` for a model and hardware pair you serve for months and can afford to compile once and cache. Choose `default` (fusion without graphs) when you serve at large batch and are compute-bound — you still want the fused kernels and the reduced HBM traffic, but the graph capture would only cost memory for a launch overhead that is already hidden. The `no-cudagraphs` variants exist because a serving engine like vLLM often wants to own graph capture itself (piecewise, around attention) while still getting autotuned kernels from Inductor — so it asks the compiler to fuse and autotune but not to wrap the result in its own graph.

### What compilation cannot fix

Set expectations honestly, because `torch.compile` is often oversold. It does not reduce the *fundamental* work: the matmul FLOPs and the HBM bytes your model must move are set by the architecture and precision, and the compiler cannot make an A100 read weights faster than its bandwidth allows. If you are genuinely compute-bound or bandwidth-bound, compilation buys you only the fusion savings, which are real but bounded. It will not fix a memory-bound decode that needs *quantization* to shrink the bytes streamed, it will not fix an attention implementation that materializes the full score matrix (that needs FlashAttention), and it will not fix a scheduler that runs tiny batches when it could run large ones. Compilation removes *overhead*; it does not change *physics*. Reach for it when a profiler shows the GPU idling on launches, and reach for a different lever when the profiler shows the GPU pinned at 95% doing real work. The discipline of measuring first — is this launch-bound, compute-bound, or bandwidth-bound? — is what separates an engineer who ships a 2x win from one who spends a week compiling a model that was never launch-bound to begin with.

### Guards, the cache-size limit, and automatic dynamic

The guard system is where compiled serving quietly succeeds or fails, so it pays to know how it behaves under shape variation. When Dynamo compiles a function, it records the assumptions it made — the exact shapes, dtypes, and Python constants — as guards. On each subsequent call it checks the guards; a pass reuses the compiled artifact, a fail triggers a recompile and stores a *new* compiled entry keyed by the new guards.

Dynamo does not recompile forever. `torch._dynamo.config.cache_size_limit` (default 8) caps how many compiled variants a single frame may accumulate. Blow past it — because, say, you feed a new batch size every step — and Dynamo logs "cache_size_limit reached" and falls back to *eager* for that frame permanently. The failure mode is nasty: your model silently reverts to the slow path you were trying to escape, and unless you are watching the logs you will not know why the compiled service is no longer fast. This is the mechanism behind most "torch.compile made it slower" reports.

There is a mitigating behavior worth understanding: **automatic dynamic shapes**. The first time a dimension changes value between calls, Dynamo assumes it was meant to be dynamic and recompiles *that dimension as symbolic* rather than as a new constant. So a size that varies should cost one extra recompile, after which a single dynamic graph covers all its values — provided the resulting graph does not itself hit a guard on something else. The relevant switches:

- `torch.compile(model, dynamic=False)` — force everything static; recompile per shape. Fastest kernels, worst recompile behavior. Fine only if you truly serve one shape.
- `torch.compile(model, dynamic=True)` — force dynamic from the first compile; one graph, generally slightly slower kernels (the compiler cannot specialize on exact sizes). Best when shapes vary widely.
- `torch.compile(model, dynamic=None)` (the default) — static first, automatic-dynamic on the first change. A reasonable default for serving if your shape set is small.
- `torch._dynamo.mark_dynamic(tensor, dim)` — surgically mark one axis dynamic (e.g. sequence length) while leaving the batch axis static so the CUDA-graph layer can still bucket it. This is usually what you want for LLM serving: dynamic sequence length, bucketed batch.

### Persisting the compile cache across restarts

Compilation is expensive, but you do not have to pay it in full on every process start. TorchInductor caches its generated kernels to disk — by default under `/tmp/torchinductor_<user>`, relocatable with `TORCHINDUCTOR_CACHE_DIR` — and an FX-graph cache (`torch._inductor.config.fx_graph_cache = True`) memoizes the graph-to-kernel lowering. A second start on the same machine and versions reuses those artifacts and compiles far faster.

For a fleet, PyTorch's *Mega-Cache* (`torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()`, PyTorch 2.5+) bundles the caches into a portable blob you can bake into your container image or fetch from object storage at boot, so a fresh pod on a fresh node warms its compile cache without recompiling from scratch:

```python
# At build/warmup time, after compiling and warming every served shape:
artifacts, info = torch.compiler.save_cache_artifacts()
open("/opt/model/compile_cache.bin", "wb").write(artifacts)

# At container start, before serving:
blob = open("/opt/model/compile_cache.bin", "rb").read()
torch.compiler.load_cache_artifacts(blob)   # skip most Inductor compilation
```

One caveat that trips people: caching accelerates *compilation*, not *CUDA-graph capture*. Graphs record device-specific memory addresses and are not serializable, so capture still runs per process. The cache shrinks the tens-of-seconds Inductor cost; the per-bucket capture cost remains and must still happen at warmup.

### AOTInductor: compiling Python out of the serving path

`torch.compile` is just-in-time — it traces and compiles inside a running Python process, and TorchDynamo stays resident to check guards. For a production server where you want *no* Python on the hot path and *no* JIT surprises, the ahead-of-time route is `torch.export` plus AOTInductor. `torch.export.export(model, args)` captures a single, whole-graph representation; `torch._inductor.aoti_compile_and_package` lowers it to a self-contained shared library you load and call from Python or C++ with none of the Dynamo machinery at runtime:

```python
import torch

ep = torch.export.export(model, (example_input,))          # whole-graph capture
path = torch._inductor.aoti_compile_and_package(ep)        # -> model.pt2 (.so inside)

# Later, in the serving process (or from C++):
runner = torch._inductor.aoti_load_package(path)
out = runner(real_input)                                    # no Dynamo, no guards
```

The trade is flexibility for determinism: AOTInductor demands an exportable graph (data-dependent control flow must be expressed with `torch.cond` / `torch.export`'s dynamic-shape API, not Python `if`), and you lose JIT specialization. In return you get a load-and-run artifact with predictable latency and no recompilation risk — which is why it underpins production serving paths and is the PyTorch-native analogue of building a TensorRT engine ahead of time. If your service value is *predictability* over peak flexibility, AOTInductor is the right tool.

## Operator fusion: collapsing the memory wall

Fusion deserves its own section because it attacks a second, distinct cost: HBM bandwidth. Cutting kernel count helps launch overhead, but *fusing* those kernels also cuts the number of times data crosses the memory bus — and at decode time, the memory bus is the wall.

Consider the tail of an MLP block: a matmul, then a bias add, then a GELU activation. In eager mode each is its own kernel. The matmul writes its output to HBM. The bias-add kernel reads that output back from HBM, adds the bias, writes it back to HBM. The GELU kernel reads it *again*, applies the activation, writes it *again*. Three kernels, and the intermediate tensor makes three full round trips to global memory even though it is never needed anywhere else.

![Before and after comparison of unfused pointwise operations against Inductor epilogue fusion collapsing them into one kernel](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-4.webp)

Inductor performs **epilogue fusion**: it folds the bias-add and the activation into the matmul kernel itself, so the result is computed, biased, and activated while it is still in registers, and only the final value is written to HBM once. Three HBM round trips become one. For memory-bound pointwise ops — and decode is dominated by them once you exclude the big matmuls — that is close to a 3x reduction in traffic for that fused region.

The fusion taxonomy is worth internalizing because it tells you what the compiler *can* and *cannot* do for free:

- **Pointwise + pointwise.** Element-wise ops chained together (bias, activation, residual add, dropout-at-eval, scaling) fuse trivially into one kernel. This is the bread and butter of Inductor and where most of the automatic wins come from.
- **Pointwise + reduction.** A pointwise op feeding a reduction (a sum, a mean, a softmax's normalization, an RMSNorm's mean-of-squares) can fuse so the reduction consumes the pointwise output without a round trip. Layer/RMS norm is a classic fused target.
- **Epilogue fusion into matmul.** Pointwise ops *after* a matmul fold into its epilogue, as above. This is high value because matmuls are the frequent, expensive anchors.
- **What does not fuse automatically.** Two large independent matmuls do not fuse into one (they are compute-bound GEMMs with their own optimal tiling), and attention itself is special — you want a purpose-built fused attention kernel there.

That last point is the bridge to hand-written kernels. FlashAttention is *the* canonical fused attention kernel: it fuses the QK-transpose matmul, the softmax, and the value matmul into a single kernel that never materializes the full attention matrix in HBM, which is what makes long-context attention tractable. `torch.compile` will use a fused scaled-dot-product-attention backend when it can, but for the state of the art you reach for FlashAttention-2 or a Triton implementation directly. The full derivation of why fusing attention beats the memory wall — the online-softmax trick, the HBM-versus-SRAM accounting — is in [Kernel fusion and FlashAttention: beating the memory wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall). Fusion via `torch.compile` and fusion via a hand-written FlashAttention kernel are the same idea applied at two different altitudes.

### Mixing hand-written kernels with the compiler

The compiler and hand-written kernels are not either/or. When you have a Triton or CUDA kernel that beats what Inductor generates — a fused RMSNorm, a custom MoE routing kernel, a quantized matmul — you register it as a custom op so `torch.compile` treats it as an opaque, fusible-around primitive rather than tracing into it and breaking the graph:

```python
import torch

@torch.library.custom_op("myops::fused_rmsnorm", mutates_args=())
def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _triton_rmsnorm(x, weight, eps)     # your hand-written Triton kernel

@fused_rmsnorm.register_fake            # shape/dtype-only "meta" impl for tracing
def _(x, weight, eps):
    return torch.empty_like(x)
```

With the op and its fake (meta) implementation registered, Dynamo can trace *through* the surrounding model, treat your kernel as a single node, and still capture the whole thing into a CUDA graph — no graph break, your kernel on the hot path. This is the escape hatch for the cases where the compiler leaves performance on the table: keep the automatic fusion for the 95% of ops that are routine, drop in a hand-tuned kernel for the 5% that matter, and let graph capture amortize the launches of both. The two altitudes of fusion compose in one forward pass.

#### Worked example: how much HBM traffic does fusion save?

Take one MLP block of an 8B model at decode, batch 1, bf16. The intermediate activation after the up-projection has hidden dimension ~14,336, so the tensor is roughly $14{,}336 \times 2$ bytes $\approx 28$ KB per token. Unfused, the bias-add and activation each read and write that tensor: that is 4 extra HBM touches (2 reads + 2 writes) of ~28 KB, ~112 KB of avoidable traffic per block per token. Across 32 layers that is ~3.6 MB per token of pure round-trip waste on the MLP epilogue alone — and the activation and gating ops elsewhere add more. At batch 64 it scales to ~230 MB/token. None of that is compute; it is bus time that fusion simply deletes. On a 2 TB/s A100 that saved 230 MB is ~115 µs of decode time handed back per token at batch 64 — small per token, but multiplied across every token of every request in a high-throughput fleet, it is real capacity.

## CUDA graphs: capture once, replay forever

CUDA graphs are the mechanism underneath `reduce-overhead`, and it is worth understanding them directly — both because you will sometimes hand-author one, and because knowing the constraints explains every quirk of how serving stacks behave.

A CUDA graph is a recorded DAG of GPU operations — kernel launches, memory copies, their dependencies — captured once and then *replayed* as a unit. Replaying issues the entire recorded sequence to the GPU with a single call, so the per-launch CPU overhead is paid once at capture time instead of on every step. NVIDIA built the feature precisely for workloads with many small kernels where launch overhead dominates, which is decode to a tee.

The lifecycle has a fixed shape, and each phase has a cost you must budget for.

![Timeline of the CUDA graph lifecycle from cold start through warmup and capture to steady-state replay](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-7.webp)

You pay a one-time warmup and capture cost per batch bucket at startup; after that every steady-state step is a launchless replay for the life of the process. Here is the pattern by hand, which is exactly what `reduce-overhead` automates:

```python
import torch

model = load_model().eval().cuda().to(torch.bfloat16)

# 1) STATIC buffers. A captured graph reads and writes fixed memory addresses.
#    You cannot feed it a fresh tensor each step — you copy new data INTO the
#    same buffers, then replay. Shapes are frozen at capture time.
B, H = 8, 4096                      # batch bucket = 8, hidden size
static_input  = torch.zeros(B, 1, H, device="cuda", dtype=torch.bfloat16)
static_output = torch.zeros(B, 1, H, device="cuda", dtype=torch.bfloat16)

# 2) WARMUP on a side stream. Capture requires the allocator and any lazy
#    initialization to have already run, or the capture will error or record
#    the wrong thing. Three iterations is the conventional minimum.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output.copy_(model(static_input))
torch.cuda.current_stream().wait_stream(s)

# 3) CAPTURE. Everything inside the context is recorded, not executed for real.
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output.copy_(model(static_input))

# 4) REPLAY on the hot path: stage new data into the captured buffers, replay,
#    read the captured output. One launch replays the whole decode step.
def decode_step(new_token_hidden: torch.Tensor) -> torch.Tensor:
    static_input.copy_(new_token_hidden)   # in-place into the captured buffer
    g.replay()                             # single launch -> full recorded step
    return static_output.clone()
```

Read the constraints off that code, because they are the reason serving is shaped the way it is:

- **Static addresses.** The graph is bound to specific memory addresses. You feed it by copying into `static_input`, never by passing a new tensor. This is why `reduce-overhead` sometimes surprises people by returning a tensor that gets overwritten on the next step — you must `clone()` outputs you intend to keep.
- **Static shapes.** Shape is frozen at capture. A graph captured for batch 8 cannot serve batch 9. This is the single most consequential constraint for serving.
- **No CPU-side control flow, no device syncs inside the captured region.** A `.item()`, a `print(tensor)`, a data-dependent Python `if` — any of these breaks capture or forces a sync that defeats the purpose.
- **Memory cost.** The graph pins a private memory pool for its intermediate activations. Capture one graph per batch bucket and the pools add up — commonly ~1–3 GB total for a set of buckets on an 8B model. That memory is subtracted from what you could have spent on KV cache, so it is a direct trade against your maximum concurrency.

The static-shape constraint is the crux. Real traffic arrives at arbitrary batch sizes, but you can only replay a graph whose captured batch size matches. The universal solution is **bucketing plus padding**: capture graphs for a fixed set of batch sizes (say 1, 2, 4, 8, 16, 32, 64), and at each step pad the actual batch up to the nearest captured bucket, running a few wasted rows so the shape matches a graph you already have. The padding costs a little compute; the graph replay saves far more launch overhead. That trade is almost always worth it below the compute-bound crossover.

### Sharing the memory pool across buckets

Capturing one graph per bucket naively means one private memory pool per bucket, and the pools do not overlap — capture ten buckets and you may burn 10–20 GB on activation scratch you will never use simultaneously. Since only one graph replays at a time, the pools *can* be shared. PyTorch exposes this through a shared pool handle:

```python
import torch

pool = torch.cuda.graph_pool_handle()      # one pool shared by all bucket graphs
graphs = {}
for b in (1, 2, 4, 8, 16, 32, 64):
    static_in[b], static_out[b] = make_static_buffers(b)
    warmup(model, static_in[b])            # 3 iters on a side stream, per bucket
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):   # reuse the same underlying pool
        static_out[b].copy_(model(static_in[b]))
    graphs[b] = g
```

The higher-level `torch.cuda.make_graphed_callables` wraps this pattern for you, and it is essentially what `reduce-overhead` and vLLM do internally: capture a family of graphs against a shared pool, then dispatch to the one matching the padded batch. Pool sharing is why a serving engine can afford a dozen buckets for a couple of gigabytes total instead of tens of gigabytes — and that couple of gigabytes is still memory you are not giving to the KV cache, which is the real opportunity cost.

### Piecewise graphs: why attention stays outside the capture

There is a subtlety that explains vLLM V1's design and every "why can't the whole model be one graph" question. A CUDA graph freezes shapes — but in continuous batching, *the attention computation's shape is not fixed even for a fixed batch size*, because each sequence in the batch has a different KV-cache length (context so far). Sequence A might have 40 tokens of context, sequence B 3,000. The attention kernel's work depends on those lengths, which change every step as sequences grow. You cannot bake that into a static graph.

The resolution is a **piecewise CUDA graph**: capture the parts of the model whose shapes *are* fixed given the batch bucket — the QKV/output/MLP projections, the norms, the residual adds, the activations, which together are the overwhelming majority of the kernel count — and leave *attention* outside the graph as a normal kernel call that receives the current KV lengths at runtime. The graph is split into segments around each attention op; between segments, the engine invokes the flexible attention kernel eagerly. You keep the launch-overhead win on the hundreds of small projection and pointwise kernels while preserving attention's shape flexibility. This is precisely how vLLM V1 combines piecewise `torch.compile` with CUDA graphs, and it is why "just capture the whole forward pass" is not how production LLM serving works. The launch-bound population — the cheap, numerous ops — is captured; the shape-varying, compute-meaningful attention stays dynamic.

### Common capture failures and how to read them

Hand-authoring graphs, or debugging an engine that captures them, you will meet a small set of errors that all trace back to the constraints above. Knowing the mapping saves hours:

- **`operation not permitted when stream is capturing` / `cudaErrorStreamCaptureUnsupported`.** Something inside the capture region did a disallowed action — most often a synchronizing call (a `.item()`, a `.cpu()`, a Python-side shape read, a `print(tensor)`), or an allocation outside the graph's pool. Remove the sync; move any host-side logic out of the captured step.
- **`illegal memory access` on replay (but not on the warmup run).** The classic cause is that you returned a reference to `static_output` and then read it *after* the next `replay()` overwrote it — the graph writes to the same address every time. Clone outputs you keep. It can also mean an input tensor moved: you passed a fresh tensor instead of copying into the captured `static_input`.
- **Wrong numbers, no error.** The graph recorded stale buffer contents because you skipped warmup, or captured before lazy initialization ran. Always do the 3-iteration warmup on a side stream before capture.
- **OOM at capture time, not at inference.** Each bucket's pool is allocated during capture; if you capture many large buckets without a shared pool handle, you exhaust memory at startup. Share the pool, or capture fewer buckets.

The through-line is that every capture error is the graph enforcing "static shapes, static addresses, no host interaction." Read the error as a pointer to which of those three rules you broke.

## CUDA graphs in a real serving loop: vLLM

You will rarely hand-author graphs in production — you will configure an engine that does it for you. vLLM is the reference implementation, and its behavior is exactly the bucketing-and-padding scheme above, wired into the scheduler.

![Branching graph routing an incoming batch through a shape dispatcher to captured graphs by bucket, with an eager fallback that can recompile](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-6.webp)

Each decode step, vLLM's scheduler forms a batch, looks up the smallest captured graph whose batch size covers the (padded) request count, and replays it. A shape that no bucket covers falls back to eager execution — and with `torch.compile` in the loop, an uncovered shape can trigger a recompile, which is the +200 ms stall you see as a latency spike. The knobs:

```python
from vllm import LLM

# enforce_eager=False (the DEFAULT) enables CUDA-graph capture at startup.
# Set it to True only to debug or when memory for graph pools is too tight.
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    enforce_eager=False,                      # False = capture CUDA graphs
    gpu_memory_utilization=0.90,
    # vLLM V1 uses piecewise torch.compile + CUDA graphs. The compilation
    # config controls which batch sizes get a captured graph. Fewer, larger
    # buckets = less capture memory + faster startup, but more padding waste.
    compilation_config={
        "level": 3,
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 48, 64, 128, 256],
    },
)
```

The equivalent server-side flags — field names have shifted across vLLM versions, so check yours:

```bash
# Default startup already captures CUDA graphs. To DISABLE for debugging:
vllm serve meta-llama/Llama-3.1-8B-Instruct --enforce-eager

# To tune capture buckets and compilation level (vLLM V1 style):
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 --gpu-memory-utilization 0.90 \
  -O '{"level": 3, "cudagraph_capture_sizes": [1,2,4,8,16,32,64,128,256]}'
```

Three operational realities that bite teams running vLLM:

- **Startup is slower with graphs on.** Capturing a graph per bucket at boot adds seconds to minutes to startup and consumes GPU memory for the pools. On a fleet that autoscales aggressively or restarts pods often, that cold-start cost is a real availability concern — you may want fewer buckets or a warm pool of pre-started replicas.
- **`enforce_eager=True` is your fast escape hatch, not your steady state.** It disables both graphs and compilation, so it is invaluable for isolating a bug ("does the problem persist in eager?") but it leaves the low-batch launch overhead on the table. Do not ship it as a permanent setting because a compilation issue was annoying to debug.
- **Bucket choice is a real tuning decision.** Too many buckets wastes capture time and memory; too few forces more padding (running batch 40 as batch 64 wastes 24 rows of compute every step). Match your buckets to your actual batch-size distribution — capture what you serve.

For how these graphs interact with continuous batching and the paged KV cache that dominates vLLM's memory, see [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) and the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive). The graph capture sits *underneath* the scheduler: the scheduler decides the batch, the graph replays it.

### Tuning buckets from your traffic, and watching them work

Bucket selection should be data-driven, not guessed. Export the batch-size histogram from a day of production traffic — vLLM's Prometheus metrics expose the running batch size and the number of requests in each scheduler step — and place your capture sizes where the mass is. If 80% of your decode steps run at batch 4–24, capture 4, 8, 12, 16, 24, 32 densely and one or two large buckets (64, 128) for spikes; do not waste capture memory on sizes you almost never hit. The padding waste for a step is `(bucket − actual) / bucket`, so tight bucket spacing where traffic concentrates directly cuts wasted compute.

To confirm the graphs are actually helping, watch three signals together. **GPU utilization** should jump at low batch after enabling capture — that is the idle bubbles disappearing. **TPOT p50 and p99** should both fall, and p99 especially should stop showing periodic spikes if you have also killed recompiles. **Startup time and steady-state memory** should rise by the capture cost — if they did not, capture probably is not happening (check that `enforce_eager` is not accidentally `True`). A useful negative test: flip `enforce_eager=True` for one canary replica and compare. If its low-batch TPOT is meaningfully worse and its GPU utilization lower, the graphs are earning their keep; if the two replicas are indistinguishable, you were already compute-bound and the capture cost is buying you nothing — a signal to spend the memory on KV cache instead.

### Warmup is a deployment step, not a code comment

The single most common way this optimization causes a production incident is a cold pod serving a live request before it has compiled and captured. The compilation and per-bucket capture cost — tens of seconds to a few minutes — must be paid *before* the pod is marked ready. Treat warmup as an explicit phase:

```python
# Startup: compile + capture EVERY served shape before readiness passes.
def warmup(engine, buckets=(1, 2, 4, 8, 16, 32, 64), seq_lens=(128, 512, 2048)):
    for b in buckets:
        for s in seq_lens:
            engine.generate(dummy_batch(b, s), max_tokens=4)  # trigger compile+capture
    torch.cuda.synchronize()

warmup(engine)
mark_ready()          # only now does the Kubernetes readiness probe pass
```

Wire this so the readiness probe fails until warmup completes, and the load balancer will not route to a half-warm pod. On an autoscaling fleet, factor the warmup duration into your scale-up lead time — if a new replica takes 90 seconds to warm, your autoscaler must react 90 seconds before you need the capacity, or the burst hits cold pods. This is the operational tax of compiled serving, and it is entirely manageable once you stop treating warmup as something that "just happens on the first request."

## Recompilation storms and how to debug them

Everything above assumes you compile and capture *once* and then replay forever. The catastrophic failure mode is when that assumption breaks and you recompile or graph-break repeatedly on the hot path. A recompilation storm turns a fast compiled model into something *slower* than eager, because you pay full compilation cost over and over while also carrying compiled-path bookkeeping. This is the number-one way `torch.compile` regresses a production service, and it is almost always caused by a small, findable set of guard failures.

![Matrix of recompilation and graph-break pitfalls listing the trigger, symptom, TORCH_LOGS signal, and fix for each](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-8.webp)

The single most useful diagnostic is `TORCH_LOGS`. It tells you, in plain text, every time Dynamo recompiles or breaks a graph and *why* the guard failed:

```bash
# Log every recompilation and the guard that triggered it:
TORCH_LOGS="recompiles" python serve.py

# Broader view — recompiles, graph breaks, and the guards themselves:
TORCH_LOGS="recompiles,graph_breaks,guards" python serve.py

# Turn recompilation from silent-slowdown into a loud error while testing:
#   torch._dynamo.config.error_on_recompile = True   (raises on 2nd compile)
```

A `recompiles` line names the offending property, for example that a tensor whose size was `[8, 512]` at compile is now `[8, 513]`. That points you straight at a dynamic shape. The fixes, in order of preference:

- **Dynamic sequence length → mark it dynamic or pad.** If the sequence dimension legitimately varies, tell Dynamo up front with `torch.compile(model, dynamic=True)` or pin the specific axis with `torch._dynamo.mark_dynamic(input, dim)`. That produces one graph that generalizes over that dimension instead of one graph per length. Alternatively, pad to bucketed lengths so only a handful of shapes ever occur — the same bucketing strategy the CUDA-graph layer already uses.
- **New input dtype → fix it upstream.** A guard failure on dtype (fp16 on one request, bf16 on the next) means your preprocessing is inconsistent. Normalize the dtype before the model, not inside it.
- **Data-dependent Python branch → rewrite it out of the graph.** An `if x.sum() > 0:` on a tensor value forces a graph break because the branch cannot be traced statically. Replace it with a `torch.where` / masked computation, or hoist the decision out of the compiled region.
- **A `.item()` or `.cpu()` on the hot path → keep tensors on device.** Any op that pulls a value back to the CPU forces a device sync and a graph break, and inside a captured region it is fatal. Sampling logic that calls `.item()` per step is a common culprit; vectorize it to stay on the GPU.

Programmatically, `torch._dynamo.explain(model)(example_input)` returns a report of graph breaks and their reasons before you ever serve traffic — run it in CI on a representative input and fail the build if the break count regresses. The discipline that keeps compiled serving fast is boring but effective: compile with `fullgraph=True` in testing, run `explain` in CI, watch `recompiles` in staging, and pin or pad every shape that varies. Do that and the storm never happens.

#### Worked example: reading a recompilation storm

Here is a real-shaped incident. A team ships a compiled Llama service; p50 TPOT is great in the load test but p99 in production is 8x worse and creeps up over the first few minutes, then the whole service degrades to eager-level latency and stays there. The load test used a fixed batch of 16; production traffic arrives at every batch size from 1 to 40 as continuous batching forms and drains batches.

Turn on `TORCH_LOGS="recompiles"` in staging and the story is immediate: a `recompiles` line for batch 1, then batch 2, batch 3, batch 5, batch 7 — a fresh compile every time a new batch size appears, each a multi-second stall that shows up as a p99 spike. After the eighth distinct size, `cache_size_limit reached` appears and the frame falls back to eager for good, which is the permanent degradation. Two fixes together resolve it: (1) `torch._dynamo.mark_dynamic(input_ids, 0)` to make the batch axis symbolic so one graph covers a range of sizes instead of one-per-size, and (2) bucket-and-pad the batch to a small fixed set (1, 2, 4, 8, 16, 32) so at most six shapes ever reach the model and the CUDA-graph layer has a bucket for each. The storm stops, p99 collapses back to p50, and it never regresses because CI now asserts a recompile count of zero on the representative shape sweep. The lesson: a compiled service must be *tested against the shape distribution it will actually see*, not a single convenient shape.

## Measuring it: a decode benchmark harness

You cannot manage what you do not measure, and the metric that matters here is TPOT — time per output token, the steady-state decode latency — across the batch sizes you actually serve. Here is a harness that measures it correctly, using CUDA events so you time GPU work and not Python:

```python
import torch

@torch.inference_mode()
def bench_tpot(step_fn, n_warmup: int = 20, n_iters: int = 200):
    """Measure ms/token and tok/s for one decode step function."""
    for _ in range(n_warmup):          # warm up compile + capture + caches
        step_fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        step_fn()
    end.record()
    torch.cuda.synchronize()           # block until GPU actually finishes

    ms_per_token = start.elapsed_time(end) / n_iters
    return ms_per_token, 1000.0 / ms_per_token   # (TPOT ms, tokens/s per stream)

# Compare eager vs compiled across the batch sizes you will serve:
for b in (1, 8, 64):
    eager_step    = make_decode_step(model,    batch=b)
    compiled_step = make_decode_step(compiled, batch=b)
    e_ms, e_tps = bench_tpot(eager_step)
    c_ms, c_tps = bench_tpot(compiled_step)
    print(f"batch={b:>3}  eager {e_ms:6.2f} ms  compiled {c_ms:6.2f} ms  "
          f"speedup {e_ms / c_ms:4.2f}x  (aggregate {c_tps * b:7.0f} tok/s)")
```

Three measurement mistakes to avoid. First, **always warm up** — the first compiled calls include compilation and the first captured calls include capture; timing them makes the compiled path look catastrophically slow. Second, **synchronize before you read the clock**, or you time launch enqueue, not execution, and the async overlap makes eager look artificially fast. Third, **measure at your real batch distribution** — a speedup at batch 1 says nothing about batch 64, because you may already be compute-bound there. Pair TPOT with a GPU-utilization read (from `nvidia-smi dmon` or the DCGM exporter) so you can see the launch-bound signature directly: low utilization with low batch is the tell. For how TPOT, TTFT, and p99 fit into an SLO, see [Model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics).

**Do not conflate prefill and decode.** These techniques help *decode*, not *prefill*, and confusing the two leads to wrong conclusions. Prefill processes the whole prompt at once — hundreds or thousands of tokens through each kernel — so its kernels have plenty of work, it is compute-bound, and launch overhead is already hidden. Its sequence length also varies per request and is large, so it is a poor fit for static graph capture. That is why serving engines typically capture graphs for *decode* steps and run prefill eagerly (or under a shape-dynamic compiled path). Concretely: `torch.compile` and CUDA graphs move your TPOT (the decode metric) and barely touch your TTFT (dominated by prefill). If your SLA pain is TTFT, this post is the wrong lever — look at chunked prefill, prefix caching, and prefill/decode disaggregation instead. If your pain is TPOT and per-token cost at small-to-moderate batch, you are in the right place. Measuring TTFT to evaluate a decode optimization is a common and expensive mistake.

The representative before→after numbers below are what you should expect on an 8B-class model in bf16. They are illustrative figures consistent with the published trends from gpt-fast and vLLM, not a single benchmark run — your exact numbers depend on model, kernels, and framework version, so measure your own. The *shape* of the table is the durable lesson.

| Hardware | Batch | Mode | TPOT (ms) | Aggregate tok/s | GPU util |
|---|---|---|---|---|---|
| A100-80GB | 1 | eager | 26 | 38 | ~30% |
| A100-80GB | 1 | compile + graph | 13 | 77 | ~75% |
| A100-80GB | 8 | eager | 30 | 267 | ~55% |
| A100-80GB | 8 | compile + graph | 18 | 444 | ~85% |
| A100-80GB | 64 | eager | 55 | 1,163 | ~88% |
| A100-80GB | 64 | compile + graph | 48 | 1,333 | ~93% |
| H100-80GB | 1 | eager | 18 | 56 | ~25% |
| H100-80GB | 1 | compile + graph | 8.5 | 118 | ~70% |
| H100-80GB | 64 | eager | 30 | 2,133 | ~92% |
| H100-80GB | 64 | compile + graph | 26 | 2,462 | ~95% |

Read the pattern, not the digits. The speedup is largest at batch 1 (roughly 2x), shrinks by batch 8 (~1.6x), and is marginal by batch 64 (~1.15x) — because at batch 64 you have crossed $b^\*$ into the compute-bound regime where launch overhead was already hidden. And the low-batch win is *bigger on the H100* than the A100 (2.1x vs 2.0x, with the H100's eager utilization even lower), exactly as the fixed-launch-cost argument predicts: the faster the GPU, the more of the idle time was launch overhead waiting to be reclaimed.

#### Worked example: what launch overhead costs you in dollars

Throughput is cost. Take the A100-80GB rows above and a representative on-demand price of \$2.00/GPU-hour (use your own contract rate — this is illustrative). Cost per million output tokens is just the GPU hourly rate divided by aggregate tokens/hour:

- **Batch 8, eager:** 267 tok/s → 0.96M tok/hr → \$2.00 / 0.96 ≈ **\$2.08 per 1M tokens**.
- **Batch 8, compiled + graph:** 444 tok/s → 1.60M tok/hr → \$2.00 / 1.60 ≈ **\$1.25 per 1M tokens** — a 40% cost reduction from a compiler flag.
- **Batch 64, eager:** 1,163 tok/s → 4.19M tok/hr → ≈ **\$0.48 per 1M tokens**.
- **Batch 64, compiled + graph:** 1,333 tok/s → 4.80M tok/hr → ≈ **\$0.42 per 1M tokens** — a 12% reduction.

The cost lever tracks the throughput lever exactly, because on a dedicated GPU your token cost is inversely proportional to tokens/s. The takeaway is the same as the latency story from the other direction: launch-overhead removal is a large cost win when you run at small-to-moderate batch (where you were launch-bound) and a small one when you already run near the compute-bound ceiling. A latency-sensitive chat service that holds batches small to protect TTFT is exactly where the dollars are — and exactly where quantization and better batching compound the saving. For the full cost-modeling treatment, see the series' cost-management material; the point here is that the mechanics section's `max(τ, g(b))` predicts your bill, not just your latency.

## Case studies and benchmarks

**gpt-fast (PyTorch team, 2023).** The PyTorch team's "Accelerating Generative AI with PyTorch II: GPT, Fast" project built a from-scratch, PyTorch-native Llama inference implementation and stacked optimizations to reach 240+ tokens/s on Llama-7B on an A100-80GB. The instructive part for us is the *order* of the levers: `torch.compile` with CUDA graphs was the first and one of the largest single multipliers applied to the eager baseline, *before* any quantization or speculative decoding. It is the cheapest big win — a compiler flag, not a new kernel — which is why it belongs at the top of any decode-latency checklist.

**PyTorch 2 / TorchInductor (Ansel et al., ASPLOS 2024).** The formal evaluation of `torch.compile` across 180+ real models reports a geometric-mean speedup of about 2.27x for inference and 1.86x for training on an A100, achieved with Dynamo's bytecode-level graph capture plus Inductor's fusion and codegen. The paper is the authoritative citation for "compile is worth a flag" and documents the guard-and-recompile machinery that governs the failure modes above.

**vLLM CUDA graphs.** vLLM enables CUDA-graph capture by default (`enforce_eager=False`) precisely because low-batch decode is launch-bound, and its documentation attributes meaningful per-step latency reductions to graph replay — commonly in the 20–40% range at low-to-moderate batch, tapering as batch grows and the engine becomes compute-bound. vLLM V1's move to piecewise `torch.compile` plus CUDA graphs is the same two-lever combination this post describes, integrated into the scheduler and KV-cache manager.

**NVIDIA CUDA graphs.** NVIDIA's own guidance frames CUDA graphs as the fix for workloads dominated by many small kernel launches — the launch overhead is fixed per kernel, so a step made of hundreds of tiny kernels spends most of its wall-clock in launch, and graph replay is the intended remedy. That framing is exactly the $N\tau$ floor we derived: graphs turn a cost that scales with $N$ into one that does not.

**TensorRT-LLM (ahead-of-time engines).** NVIDIA's TensorRT-LLM takes the AOT position to its conclusion: it builds a heavily fused, kernel-autotuned engine ahead of time — including in-graph fusion and captured graphs — and ships a serialized engine plan you load and run. The steady-state kernels are typically the fastest available on NVIDIA hardware, at the cost of long build times and low flexibility (an engine is tied to shapes, precision, and often the exact GPU generation). It sits at the far "static, fully compiled" end of the same trade-off axis as `torch.compile`: more warmup and rigidity in exchange for the lowest steady-state overhead. The right choice when your model and shapes are frozen and you serve them for months.

**SGLang.** SGLang, another high-throughput LLM engine, combines `torch.compile`, CUDA graphs, and its RadixAttention prefix cache, and its published low-latency decode numbers come substantially from the same launch-overhead removal described here, layered on top of aggressive KV-cache reuse. It is a useful existence proof that the compiler-plus-graph path composes with cache-level optimizations rather than competing with them.

### How this composes with the rest of the stack

Launch-overhead removal is one lever among several, and it is worth being explicit about how it stacks so you sequence your work correctly:

- **With quantization.** FP8 or INT4 weights shrink $g(b)$ — the per-kernel GPU time — by reducing the bytes streamed from HBM. That *lowers* the compute-bound crossover $b^\*$, which means you stay launch-bound up to a larger batch, which means CUDA graphs help across a *wider* batch range after you quantize. Quantization and graphs are complementary, not redundant. See [Quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving).
- **With continuous batching.** A scheduler that keeps the batch full pushes you toward the compute-bound regime where the GPU is well used. But real traffic still spends time at small batch (early morning, tail sequences draining), and that is exactly when graphs rescue you. The two are partners: batching maximizes the batches you have, graphs make the small ones cheap.
- **With tensor parallelism.** Compile and capture happen per rank; each GPU captures its own shard's graph. The collective communication (all-reduce) can sit inside or outside the captured region depending on the engine, and NCCL calls have their own launch cost that graphs also amortize. Multi-GPU does not exempt you from launch overhead — it multiplies the number of processes paying it.
- **With speculative decoding.** Speculative decoding changes the shape of a step (verify $k$ draft tokens at once), which interacts with your bucketing — you must capture graphs for the speculative shapes too. It reduces the *number* of steps rather than the cost per step, so it is orthogonal to and compounds with per-step overhead removal.

The rule of thumb to carry away: quantization and fusion cut the *cost* of each step, graphs cut the *overhead* of issuing each step, batching and speculation cut the *number* of steps. A fully tuned decode path pulls all three levers.

## When to use this (and when not to)

The decision is not "always compile." It is a match between how static your serving shapes are and how much warmup latency your deployment can absorb.

![Matrix comparing eager, torch.compile, manual CUDA graphs, and vLLM cudagraphs across overhead, warmup, flexibility, and use case](/imgs/blogs/kernel-fusion-cuda-graphs-torch-compile-5.webp)

| Situation | Do this | Why |
|---|---|---|
| Small-batch LLM decode (batch 1–16), latency-sensitive | `torch.compile(mode="reduce-overhead")` or a graph-capturing engine | You are launch-bound; removing launch overhead is a ~1.5–2x free win |
| Serving through vLLM / TGI | Leave graph capture on (`enforce_eager=False`), tune buckets | The engine already does the right thing; just don't disable it |
| Large-batch throughput serving (batch ≥ 64) | Compile for fusion, but expect small graph gains | You are compute-bound; launch overhead is already hidden |
| Highly dynamic shapes you cannot bucket | Eager, or `dynamic=True` and accept fewer graphs | Recompilation storms will erase the win |
| Frequently restarting / aggressively autoscaling pods | Weigh warmup cost; consider fewer buckets or warm pools | Per-bucket capture at boot can dominate a short pod lifetime |
| Debugging a correctness or perf regression | `enforce_eager=True` temporarily | Isolate whether the bug is in the compiled path |

Say it plainly: **do not reach for CUDA graphs or `torch.compile` when your shapes are genuinely unbucketable, when you are already compute-bound at your serving batch size, or when your process lifetime is so short that warmup dominates.** A batch-serving pipeline running at batch 256 for offline generation will see almost nothing from graph capture and will pay its warmup and memory cost for no return — spend that engineering effort on quantization or better batching instead. And never let the 30-second-to-2-minute compilation land on a live request: warm every served shape before the readiness probe passes, or you will ship minute-long TTFT spikes on cold pods. The technique is a scalpel for launch-bound decode, not a blanket "make it fast" button.

#### Worked example: should this chat service compile?

A concrete decision. You run Llama-3.1-8B for an interactive chat product on A100-80GB. Requirements: p95 TPOT under 40 ms so streaming feels smooth, TTFT under 500 ms, and you hold batches small (typically 4–20 concurrent decodes) to protect latency rather than maximize throughput. Should you turn on compilation and graphs?

Walk the checklist. **Are you launch-bound?** At batch 4–20 on an A100, almost certainly yes — a profiler trace shows GPU utilization around 40–55% with gaps in the CUDA HW row. That alone says there is a ~1.5–1.8x TPOT win available. **Are your shapes bucketable?** Batch size, yes — bucket to 1, 2, 4, 8, 16, 24, 32. Sequence length varies, so `mark_dynamic` the sequence axis and keep the batch axis bucketed for the graph layer. **Can you afford warmup?** The service is long-lived and does not restart per request, so a 60–90 second startup warmup is fine; gate it behind the readiness probe. **Memory?** The bucket graphs cost ~1–2 GB with a shared pool, which you can spare from KV cache at this batch size. **Verdict:** yes — enable `enforce_eager=False` with those capture sizes, `mark_dynamic` on the sequence dimension, warm at startup, and assert zero recompiles in CI. Expected result, from the benchmark table: TPOT roughly halves at batch 4–8, comfortably clearing the 40 ms p95 target, and your per-token cost drops ~30–40%. Now flip one variable: if this were an *offline batch* job running at batch 256 to summarize a corpus overnight, every answer reverses — you are compute-bound, warmup is a large fraction of a short job, and you would skip graphs and reach for quantization. Same model, opposite decision, decided entirely by where you sit relative to $b^\*$.

The good news is that the two levers compose cleanly with the rest of the serving toolbox. Fusion and graphs reduce per-step overhead; quantization reduces the per-step *memory* you must stream; continuous batching keeps the batch full so you stay near the compute-bound regime where the GPU is well used. They are independent trades on the same SLO triangle, and a well-tuned decode path uses all of them.

## Key takeaways

- **Small-batch decode is launch-bound, not compute-bound.** Step time is $N \cdot \max(\tau, g(b))$; below the crossover batch $b^\*$ the CPU's per-launch cost $\tau$ dominates and the GPU idles. Prove it with $N\tau$ and a utilization read before you optimize.
- **`torch.compile` is a four-stage pipeline, not a black box.** Dynamo captures and guards, AOTAutograd normalizes, Inductor fuses and generates Triton, and `reduce-overhead` wraps the step in a CUDA graph. Each stage removes a distinct overhead.
- **Fusion attacks two costs at once.** It cuts kernel count (launch overhead) and HBM round trips (bandwidth). Epilogue fusion folds bias and activation into the matmul; pointwise chains collapse to one kernel.
- **CUDA graphs turn $N$ launches into one replay** — but only for static shapes and static addresses, which is why serving buckets and pads batch sizes, and why captured graphs cost 1–3 GB of memory you trade against KV cache.
- **The win is largest at batch 1 and on faster GPUs**, and shrinks toward nothing as you cross into the compute-bound regime at large batch. Measure at your real batch distribution.
- **Recompilation storms are the top regression.** Dynamic shapes, dtype changes, data-dependent branches, and `.item()` syncs all break guards. `TORCH_LOGS="recompiles"` names the cause; pin shapes, mark dynamic axes, or pad.
- **Warmup is a hard cost, not an afterthought.** Compilation and capture take seconds to minutes per shape; do it at startup before traffic, never on a live request.
- **Use `enforce_eager=True` to debug, never to ship.** It is the fast way to isolate a compiled-path bug and the slow way to serve production traffic.
- **These levers help decode, not prefill.** They move TPOT and per-token cost, not TTFT. If your SLA pain is TTFT, reach for chunked prefill, prefix caching, or prefill/decode disaggregation instead.
- **The techniques compose.** Quantization and fusion cut per-step cost, graphs cut per-step overhead, batching and speculation cut the number of steps. Sequence them; do not treat them as alternatives.
- **Bucket from traffic, not from a round number.** Export the batch-size histogram, capture where the mass is, and confirm the win with a `enforce_eager=True` canary — if the canary matches, you were compute-bound and the capture memory is better spent on KV cache.

## Further reading

- **"Accelerating Generative AI with PyTorch II: GPT, Fast"** — PyTorch team, 2023. The gpt-fast writeup: `torch.compile` + CUDA graphs as the first and largest lever on Llama-7B decode. (pytorch.org blog)
- **"PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"** — Jason Ansel et al., ASPLOS 2024. The authoritative `torch.compile` / TorchInductor paper; ~2.27x geomean inference speedup on A100.
- **NVIDIA CUDA C++ Programming Guide and "Getting Started with CUDA Graphs"** — the mechanics of graph capture, replay, and why launch overhead dominates small-kernel workloads.
- **FlashAttention-2** — Tri Dao, 2023 — the reference fused attention kernel; the hand-authored complement to Inductor's automatic fusion.
- **vLLM documentation** — engine arguments (`enforce_eager`, compilation config, CUDA-graph capture sizes) and the V1 piecewise-compile design notes.
- Within this series: [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving), [Why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different), [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive), [Custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference), and the memory-wall companion [Kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall).
