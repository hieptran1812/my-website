---
title: "The Inductor CPU Backend: Debugging and Profiling Compiled CPU Inference"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Not every AI service has a GPU. When your model runs on CPU, torch.compile still helps — it emits vectorized C++ and OpenMP through the Inductor CPU backend. This post shows how that backend works, how to debug it when it produces the wrong answer, and how to profile it when it is slow."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "torch-compile",
    "inductor",
    "cpu",
    "profiling",
    "pytorch",
    "vectorization",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 28
---

Most of this series lives on a GPU, because most of the expensive AI in the world does. But a large and quiet fraction of inference runs on CPUs — embedding services that fan out to hundreds of cheap cores, on-device models, feature transformers wedged between a database and a ranker, batch jobs that would cost more to schedule on a GPU than they save. On those services the four wastes from the [series intro](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) still apply — you can still be host-bound on Python dispatch, still be memory-bound streaming activations through cache, still do redundant work that a fused kernel would have collapsed — and `torch.compile` still helps, through a backend most GPU engineers never look at: the **Inductor CPU backend**, which lowers your model to vectorized C++ with OpenMP threading and hands it to your system compiler.

The trouble is that debugging and profiling compiled CPU code is a genuinely different craft from the GPU story the rest of the track told. There is no `nsys` timeline, no `ncu` occupancy, no CUDA graph. Instead there is generated C++ you can actually open and read, a minifier that bisects a miscompiled graph down to one bad operator, and a profiler view where the whole question is whether your hot loop got vectorized and whether OpenMP actually spread it across cores. This post is the CPU half of the compile track: how the Inductor CPU backend produces its code, how to debug it when the numbers come out wrong, and how to profile it when the latency comes out high.

![a vertical stack showing the inductor cpu lowering path from an fx graph down through cpp and openmp code generation with simd vectorization to a compiled shared object that runs](/imgs/blogs/inductor-cpu-backend-debugging-and-profiling-1.webp)

By the end you will be able to: turn on `torch.compile` for a CPU model and confirm it picked the C++/OpenMP backend; read the generated `.cpp` to see whether your pointwise chain fused and whether the inner loop uses `at::vec` SIMD; use `TORCH_COMPILE_DEBUG=1` and the minifier to isolate a miscompiled operator to a minimal reproducer; profile the compiled model to decide whether it is host-bound (compile helps) or compute-bound (you need threads and vectorization, or a different tool); and make the honest call about when to reach for ONNX Runtime or OpenVINO instead. The running example is a small Transformer encoder and an embedding model served on CPU — the shapes an actual CPU service handles.

## How the Inductor CPU backend generates code

When you call `torch.compile(model)` on a model whose tensors live on the CPU, the front of the stack is identical to the GPU path: TorchDynamo traces your Python into an FX graph, inserts guards, and AOTAutograd functionalizes it. What changes is the backend. Inductor has two code generators — a Triton generator for CUDA, and a **C++ generator for CPU** — and it selects the C++ one automatically for CPU tensors. That generator lowers the fused graph to a `.cpp` file, compiles it with your system compiler (`g++` or `clang++`) into a shared object, and loads it back with `ctypes`. The compiled artifact is cached on disk under `TORCHINDUCTOR_CACHE_DIR` exactly like the Triton cache, so the second process to compile the same graph reuses the `.so`.

The generated C++ does three things that plain eager execution does not. First, it **fuses pointwise chains** the same way the GPU backend does: a residual add, a `gelu`, and a `layer_norm` that eager would run as three separate dispatched operators — three passes over the activation tensor through the cache hierarchy — become one C++ loop that computes all three while the data is still in registers. Second, it **vectorizes** the inner loop with SIMD, emitting `at::vec::Vectorized<float>` operations that map to AVX2 (8 floats per instruction) or AVX-512 (16 floats) on x86, or NEON on ARM. Third, it **parallelizes** the outer loop with OpenMP `#pragma omp parallel for`, spreading the work across the cores you gave it. Fusion attacks redundant memory traffic; vectorization and threading attack the compute. Miss any one of the three and the compile underperforms, which is exactly what the profiling section teaches you to detect.

There is also a CPU `mode="max-autotune"`, and it means something narrower than on the GPU: rather than benchmarking many matmul templates, it turns on Inductor's C++ GEMM template and autotunes the fused loops' tiling and thread partitioning, occasionally beating the oneDNN default on unusual shapes. As on the GPU it costs far more compile time for a usually-small runtime change, so it is an offline-serving lever, not a default — profile before and after to decide whether the extra `g++` minutes bought anything at runtime.

Here is the whole thing made concrete. Compile a small model on CPU and dump what Inductor wrote:

```python
import torch

def block(x, w, b):
    y = torch.addmm(b, x, w)      # a GEMM (stays on the CPU BLAS, e.g. MKL/oneDNN)
    y = torch.nn.functional.gelu(y)   # pointwise
    return y + x[:, : y.shape[1]]     # residual add, pointwise

model = torch.compile(block)
x = torch.randn(256, 768)
w = torch.randn(768, 768)
b = torch.randn(768)
model(x, w, b)   # first call compiles; writes a .cpp under the inductor cache
```

Set `TORCH_LOGS="output_code"` and Inductor prints the C++ it generated for the fused pointwise part — the `gelu` and the residual add collapsed into one vectorized, threaded loop:

```cpp
extern "C" void kernel(const float* in_ptr0, const float* in_ptr1, float* out_ptr0) {
    #pragma omp parallel for
    for (long i0 = 0; i0 < 196608L; i0 += 16L) {        // 256*768 = 196608 elements
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + i0);   // gemm output, load 16
        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + i0);   // residual, load 16
        auto tmp2 = tmp0 * (0.5f * (1.0f + (tmp0 * 0.7071f).erf()));   // GELU, in registers
        auto tmp3 = tmp2 + tmp1;                                       // residual add, fused
        tmp3.store(out_ptr0 + i0);                                     // store 16
    }
}
```

Read it the way you read a fused Triton kernel in [profiling compiled code](/blog/machine-learning/performance-engineering/profiling-compiled-code): two `loadu`s at the top, one `store` at the bottom, everything in between in registers, the loop striding by 16 because AVX-512 processes 16 floats per instruction, and a `#pragma omp parallel for` spreading the iterations across cores. That single loop is the fusion, the vectorization, and the threading, all visible in source you can read. If any of the three is missing — a scalar loop with no `at::vec`, or no `omp` pragma — you have found your problem before you have run a single profile.

## Eager versus compiled on the CPU

The intuition for why this is faster is worth making precise, because on the CPU the win comes from a different mix than on the GPU.

![a two column before and after contrasting eager cpu execution of per operator dispatch and scalar loops against compiled cpu execution with fused vectorized loops running across openmp threads](/imgs/blogs/inductor-cpu-backend-debugging-and-profiling-2.webp)

In eager mode, each operator is a separate dispatch through the PyTorch dispatcher — a virtual call, a kernel lookup, a launch of an ATen CPU kernel — and each ATen kernel makes its own pass over the tensor. For a chain of small pointwise operators on a modest tensor, the *dispatch overhead itself* can rival the arithmetic, and the repeated passes over the same data blow the L2 cache. That is the CPU form of being **host-bound**: the Python-and-dispatch machinery, not the math, sets your latency. Compiled code erases the per-operator dispatch (one C++ call runs the whole fused chain), erases the redundant cache passes (one pass, data kept in registers), and vectorizes plus threads the arithmetic that remains. On an elementwise-heavy model that is a large win; on a model that is already one big matmul, it is almost nothing, because the matmul was already a single well-tuned BLAS call and there was never any dispatch overhead or fusion opportunity to reclaim.

Numbers make the split clear. Measured on an Intel Xeon (Ice Lake, AVX-512, 16 cores bound), a 6-layer Transformer encoder at batch 8, sequence 128, the eager model runs a p50 of 24.1 ms; `torch.compile` brings it to 14.8 ms, a 39% win. Almost all of that comes from the pointwise fusion and vectorization of the normalization, activation, and residual glue — the `addmm` GEMMs stay on oneDNN and barely move. Push the batch to 64 and the picture shifts: the model becomes compute-bound on the GEMMs, the fusible glue is a smaller fraction of the total, and the compile win shrinks to 12%. Same code, same compile call; the win depends entirely on whether the model was spending its time on the thing compile can fix.

### How Inductor structures a generated loop: tiling, tails, and reductions

The single fused loop above is the easy case — a pure pointwise chain where every element is independent, so Inductor emits one flat `#pragma omp parallel for` and strides by the vector width. Real models are not all pointwise, and the two structures Inductor generates for the harder cases are worth recognizing, because their failure modes differ.

The first is **tiling with a masked tail**. A vectorized loop that strides by 16 only works cleanly when the element count is a multiple of 16; the leftover elements at the end — the *tail* — need special handling. Inductor emits a main vectorized loop over the aligned bulk and then a masked or scalar epilogue for the remainder. You will see it in the generated C++ as a second, shorter loop after the main one, and it matters for a specific reason: if your hidden dimension is an awkward size (say 130 instead of 128), a disproportionate fraction of the work falls into the slow scalar tail, and the kernel underperforms for a reason that is invisible unless you read the generated code. Padding hidden dimensions to multiples of the vector width is a real CPU optimization, and this is why.

The second is the **reduction kernel**. Operations like `layer_norm`, `softmax`, `mean`, and `sum` reduce along an axis, and a reduction cannot be trivially vectorized element-by-element the way a pointwise op can — you have to accumulate. Inductor generates two flavors, the same categories you learned to read in the [compiled-code post](/blog/machine-learning/performance-engineering/profiling-compiled-code): a plain reduction that streams the reduced axis, and a *persistent* reduction (`cpp_fused_..._per` in the profiler) that keeps the whole reduced dimension in a register file when it fits — the common case for a layer-norm over a 768-wide hidden dim. The persistent form is much faster because it avoids re-reading the data, but Inductor only generates it when it can prove the reduction dimension is small and contiguous. A layer-norm over a strided or transposed axis defeats the persistent form and falls back to the streaming reduction, which is exactly the kind of slowdown the worked example below hunts down.

There is also a mode toggle worth knowing: `torch._inductor.config.cpp_wrapper = True`. By default Inductor's CPU path uses a Python wrapper to call the generated kernels, which reintroduces a little per-call Python overhead — fine when the kernels are large, but a measurable tax on a model of many tiny kernels. The `cpp_wrapper` mode generates a C++ wrapper too, so the entire compiled subgraph runs with no Python in the hot path, which is the CPU analogue of what CUDA graphs do for GPU launch overhead. On a batch-1, small-tensor CPU service where dispatch dominates, turning it on can shave another 10–20% off the already-compiled latency. It is not on by default because it complicates debugging — the wrapper is harder to read than the Python one — so reach for it once correctness is settled and you are chasing the last of the host overhead.

## Debugging correctness: when the compiled model is wrong

The failure that CPU compilation adds over the GPU story is a *wrong answer*. A miscompiled operator, a vectorization bug, a lowering that mishandles an edge case — these are rare, but when they happen the model does not crash, it returns numbers that are subtly off, and you need a disciplined way to find the culprit in a graph of hundreds of operators. PyTorch ships exactly that: the **minifier**.

![a dataflow where a full model reproducer fans into a bisection search over subgraphs that isolates a single miscompiled operator and merges into a minimal reproducer script](/imgs/blogs/inductor-cpu-backend-debugging-and-profiling-4.webp)

The first move when a compiled CPU model disagrees with eager is to confirm the disagreement precisely and let the minifier bisect it. You set an environment variable that tells Dynamo to check the compiled output against eager after AOTAutograd and, on a mismatch, automatically bisect the graph:

```bash
# Check accuracy after AOT lowering; on failure, write a minimal repro.
TORCHINDUCTOR_REPRO_AFTER="aot" TORCHINDUCTOR_REPRO_LEVEL=4 python serve.py
```

When it trips, the minifier repeatedly cuts the graph in half, recompiles each half, and keeps the half that still reproduces the mismatch, until it has the smallest subgraph — often a single operator — that miscompiles. It writes a self-contained `repro.py` you can run, file as a bug, or work around. That bisection is the CPU analogue of the eager-vs-compiled `allclose` gate from the [CUDA graphs gotchas](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) post: the same "compare against the trusted implementation and shrink the difference" discipline, automated.

Two knobs shape what the minifier hunts. `TORCHINDUCTOR_REPRO_AFTER` chooses *where* in the stack to check — `"aot"` checks after AOTAutograd (catches Inductor lowering bugs, the common case) while `"dynamo"` checks after tracing (catches Dynamo bugs). `TORCHINDUCTOR_REPRO_LEVEL` chooses *what* counts as a failure: level 2 catches outright compile errors and crashes, level 3 catches segfaults in the generated code, and level 4 — the one you want for silent wrong answers — catches *accuracy* mismatches where the code runs fine but the numbers drift beyond tolerance. Accuracy minification is the slowest because it recompiles and runs each candidate subgraph, but it is the only mode that finds the failure that actually bites a production service: the model that returns plausible-but-wrong embeddings and quietly poisons a vector index.

It helps to know the shape of the bugs the minifier finds, because they cluster. The most common is a **vectorization miscompile** — an edge case in the masked tail or a reduction where the vectorized path and the scalar path disagree on a boundary element, usually surfacing as a tiny numerical difference at the last few elements of a dimension. The second is a **fusion-order bug**, where an in-place operation and its consumer got scheduled such that a value is read after it was overwritten — rare, but it produces large, obvious garbage rather than a small drift. The third is not a compiler bug at all but a **precision difference**: the fused kernel computes a GELU or a softmax in a different order than eager, and in float32 the reassociation changes the last bit, which is *correct* but trips a too-tight `allclose` tolerance. Distinguishing the third from the first two is why the minifier reports the actual max-difference: a `1e-6` drift is reassociation you should widen your tolerance for, while a `1e-1` drift is a real bug worth filing.

To *see* what Inductor did rather than just that it was wrong, dump the full compile debug directory:

```bash
TORCH_COMPILE_DEBUG=1 python serve.py
# writes torch_compile_debug/run_.../ with, per graph:
#   fx_graph_readable.py     -- the traced FX graph
#   output_code.py           -- the generated C++/OpenMP
#   ir_post_fusion.txt       -- every fusion group Inductor formed
```

The `output_code.py` is the same C++ you saw above; `ir_post_fusion.txt` lists exactly which operators fused into which kernels — the ground truth when a profiler row or a wrong number surprises you. Ninety percent of "the compiled CPU model is wrong or slow" investigations end here: you open the generated loop, and either the fusion you expected is absent (a graph break split it) or the vectorization is (a scalar fallback), and the file tells you which.

Here is the debug rubric as a single table — the tools, and the precise question each one answers.

![a four by three table matching cpu compile debug tools to what each one shows and the question it answers](/imgs/blogs/inductor-cpu-backend-debugging-and-profiling-3.webp)

| Tool | What it shows | Question it answers |
|---|---|---|
| `TORCH_LOGS="output_code"` | the generated C++/OpenMP | did it fuse and vectorize? |
| `TORCH_COMPILE_DEBUG=1` | full per-graph debug dir | which ops fused, what IR |
| minifier (`REPRO_AFTER`) | a minimal miscompile repro | which operator is wrong |
| `torch.profiler` (CPU) | per-kernel CPU time | where is the time going |

## Profiling compiled CPU code

Once the answer is correct, the question becomes speed, and the first thing to establish — before any optimization — is whether the model is **host-bound** or **compute-bound**, because they have opposite fixes and the profile tells you which in one read.

![a decision tree from a slow compiled cpu model branching on whether the inner loop is vectorized whether openmp threads are used and whether the model is host or compute bound down to concrete fixes](/imgs/blogs/inductor-cpu-backend-debugging-and-profiling-5.webp)

Profile the compiled model with `torch.profiler` and CPU activities, and read the table the same way you read a GPU one — top row is where the time is:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with torch.no_grad():
        for _ in range(20):
            compiled(x)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
```

```console
---------------------------------------  ------------  ------------  ------------
Name                                       Self CPU     CPU total    # of Calls
---------------------------------------  ------------  ------------  ------------
graph_0_cpp_fused_gelu_add_2                 6.812ms       6.812ms            20
aten::addmm  (oneDNN gemm)                   5.204ms       5.204ms            60
graph_0_cpp_fused_native_layer_norm_1        2.331ms       2.331ms            20
aten::copy_                                  0.402ms       0.402ms            40
---------------------------------------  ------------  ------------  ------------
Self CPU time total: 14.749ms
```

The compiled kernels carry `cpp_fused_` names — the CPU analogue of `triton_..._fused_...` — and their presence is your proof that fusion happened, exactly as on the GPU. Now the host-vs-compute question. If the top of the table is dominated by many small operators and `aten::` dispatch overhead rather than fat `cpp_fused_` kernels, you are host-bound, and compile plus fewer dispatches is the fix. If the top is one or two big `cpp_fused_` or `aten::addmm` kernels running near the core's arithmetic peak, you are compute-bound, and your levers are vectorization width and thread count — not more fusion.

Two checks turn that into a concrete diagnosis. First, **did it vectorize?** Open the generated C++ (`TORCH_LOGS="output_code"`) and confirm the hot loop uses `at::vec::Vectorized` and strides by 8 or 16, not a scalar loop striding by 1. A scalar fallback — Inductor emits one when it cannot prove the access is vectorizable — silently costs you an 8-to-16× slowdown on that loop, and it is invisible in the profiler table (the kernel just looks slow). Second, **are the threads working?** OpenMP only parallelizes if you gave it cores and the loop is big enough to be worth splitting. Control and check the thread count explicitly:

```python
import torch
torch.set_num_threads(16)          # intra-op parallelism (the omp team size)
print(torch.get_num_threads())     # confirm it took
# also set at the process level for the OpenMP runtime:
#   OMP_NUM_THREADS=16  and pin with numactl --cpunodebind=0 --membind=0
```

The trap here is **oversubscription**: if OpenMP thinks it has 32 threads but you pinned the process to 16 cores, or you run 8 replicas each grabbing all cores, the threads fight over cores and context-switch, and throughput collapses. The fix is the subject of its own post — [CPU affinity, NUMA, and threading](/blog/machine-learning/performance-engineering/cpu-affinity-numa-and-threading) — but the one-line rule is: threads per process times replicas per box should equal physical cores, and each replica should be pinned. Compilation makes the arithmetic faster; it cannot save you from thread contention you created around it.

There is a second, subtler threading knob that the profiler is the only honest way to read: PyTorch distinguishes **intra-op** parallelism (`torch.set_num_threads` — the OpenMP team size *inside* one operator, splitting a big matmul or a fused loop across cores) from **inter-op** parallelism (`torch.set_num_interop_threads` — running independent operators concurrently on separate threads). For a single-request latency-bound service you want most of your cores on intra-op parallelism so each operator finishes as fast as possible; for a throughput-bound service handling many concurrent requests you often want the opposite — one thread per operator and many requests in flight — because per-request latency matters less than keeping every core busy with useful work. Getting this backwards is a classic CPU-serving mistake: a latency service configured for throughput leaves cores idle inside each operator, and a throughput service configured for latency thrashes as every request tries to grab every core. The profiler shows it directly — an intra-op-starved run has a fat kernel whose wall time barely drops when you add cores (the loop is not being split), while an oversubscribed run shows the same kernel getting *slower* as you add concurrency. Measure both configurations under your actual traffic shape; the right answer is a property of your workload, not a universal default, and only the profile settles it.

#### Worked example: the scalar-fallback loop

An embedding service compiles a normalization-heavy encoder on an AMD EPYC (32 cores bound, AVX2) and sees only a 9% win from `torch.compile`, far below the 35% the team expected from a pointwise-heavy model. The profile shows a fat `cpp_fused_native_layer_norm_div_0` kernel eating 60% of the time — so fusion happened, but the kernel is slow. Opening the generated C++ with `TORCH_LOGS="output_code"` reveals the inner loop is *scalar*: no `at::vec`, striding by 1, because the layer-norm's reduction over a non-contiguous stride defeated Inductor's vectorizer. The fix is upstream — make the normalized dimension contiguous (a `.contiguous()` before the norm, or a layout change) so Inductor can prove the access is vectorizable — after which the same kernel regenerates with `at::vec::Vectorized<float>` striding by 8, and the block's CPU time drops from 6.8 ms to 1.9 ms. The compile was never broken; the vectorizer bailed on a stride, and only reading the generated loop revealed it. This is the CPU version of the whole series' lesson: the profiler says *which* kernel is slow, and the generated code says *why*.

## The debug-and-profile workflow, end to end

Pull it together into the loop you actually run when a compiled CPU model misbehaves — wrong or slow, the entry points differ but the tools converge.

![a left to right timeline of the cpu compile debug workflow from compiling through dumping the debug directory reading the generated cpp finding the bad loop or operator applying a fix and re profiling](/imgs/blogs/inductor-cpu-backend-debugging-and-profiling-6.webp)

Wrong answer? Set `TORCHINDUCTOR_REPRO_AFTER="aot"` and let the minifier bisect to the bad operator, then work around it or file it. Slow? Profile with CPU activities, read whether you are host-bound (fix: ensure fusion, fewer dispatches) or compute-bound (fix: vectorization and threads), then dump `TORCH_COMPILE_DEBUG=1`, open `output_code.py`, and confirm the hot loop is fused, vectorized, and threaded. Whichever branch you entered, you end in the same place — reading the generated C++ — because on the CPU that file is the ground truth. Fix, re-profile, and confirm the kernel that was slow is now fast. It is the [profile, hypothesize, fix, measure](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) loop of the whole series, with the CPU's own instruments.

#### Worked example: host-bound to fused on a laptop

A prototype ranking model runs eager on an Apple M2 (8 cores, NEON) at 41 ms p50 for a batch-1 request, and the profile is a wall of tiny `aten::` pointwise operators — classic host-bound, the dispatch overhead dwarfing the arithmetic on the small batch-1 tensors. `torch.compile(model)` fuses the pointwise chains into a few `cpp_fused_` kernels and cuts the dispatch to almost nothing; p50 falls to 23 ms, a 44% win, all of it from eliminating per-operator dispatch and cache passes rather than from faster math. The generated C++ confirms it: NEON vectorization (stride 4 for float32) and fused loops. The lesson matches the GPU intro's: batch-1, small-tensor serving is where host overhead dominates and where collapsing the op count — whether by CUDA graphs on the GPU or Inductor fusion on the CPU — pays the most.

## The compile-time bill: g++ is not free

The [break-even accounting](/blog/machine-learning/performance-engineering/profiling-compiled-code) from the compiled-code post applies on the CPU with one twist that catches people: the compile step includes invoking your *system C++ compiler*, and `g++ -O3` is genuinely slow. Where the GPU path pays its time in Triton compilation, the CPU path pays it in `g++` or `clang++` churning through the generated `.cpp`, and for a model with many distinct kernels that can be tens of seconds — sometimes longer than the Triton path it replaces. A cold compile of the 6-layer encoder above takes about 18 seconds on the Xeon, of which roughly 12 is the C++ compiler.

That makes the persistent cache even more important on CPU than on GPU. Inductor writes the compiled `.so` files to `TORCHINDUCTOR_CACHE_DIR`, and a warm cache skips the entire `g++` step — the second process to run the same graph loads the shared object in milliseconds. The operational rules are the same as the GPU story but the stakes are higher because the compiler is slower:

```bash
# Bake the CPU kernel cache into the container image, or mount it warm.
export TORCHINDUCTOR_CACHE_DIR=/mnt/warm/torchinductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
# Optionally pick a faster compiler if clang builds your kernels quicker:
export CXX=clang++
```

The break-even law says compiled wins after `$N^{*} = C / (t_e - t_c)$` steps. Plug the encoder in: cold compile `$C = 18$` s, eager step `$t_e = 24.1$` ms, compiled step `$t_c = 14.8$` ms, so `$N^{*} = 18 / 0.0093 \approx 1{,}935$` steps cold. A long-lived serving process clears that in seconds of traffic and should absolutely compile; a nightly batch that scores 1,000 rows and exits will *lose* wall-clock to the compile unless the cache is warm, at which point `$C$` drops to well under a second and the break-even collapses to a few dozen steps. The discipline is identical to the GPU case; only the constant `$C$` is bigger because it routes through your system compiler.

#### Worked example: the GEMM-bound model compile barely helped

A team compiles a wide two-layer MLP classifier — essentially two big `Linear` layers with a ReLU between — on the Xeon and sees a 4% improvement, and someone concludes "torch.compile does nothing on CPU." Run the diagnosis. The profile shows two fat `aten::addmm` (oneDNN) kernels eating 91% of the time and a single tiny `cpp_fused_relu_0`. So compile *did* fuse the only fusible op (the ReLU), and it generated correct vectorized code — there was simply almost nothing to fuse, because the model is two matmuls that oneDNN already runs near the core's peak. This is the CPU form of the [compiled-but-not-faster null result](/blog/machine-learning/performance-engineering/profiling-compiled-code): the compile had nothing to do, and 4% is the honest, correct answer, not a failure. The fix for *this* model's latency is not more compile — it is a smaller model, INT8 quantization, or a dedicated runtime — because the time is in the GEMMs, and Inductor does not replace oneDNN GEMMs by default. Knowing that in five minutes from the profile, instead of thrashing on compile flags for a day, is the whole point of measuring first.

## When to use torch.compile on CPU (and when to reach for something else)

Compiling on the CPU is a real win in a specific regime and a waste — or the wrong tool — outside it. The honest decision compares Inductor against the dedicated CPU inference runtimes.

![a three by three grid rating torch compile inductor onnx runtime and openvino cpu backends on ease of use peak speed and best fit workload](/imgs/blogs/inductor-cpu-backend-debugging-and-profiling-7.webp)

`torch.compile` with the Inductor CPU backend is the right first reach when you are staying in PyTorch, your model is pointwise-heavy (normalization, activations, residuals, elementwise feature transforms) so there is fusion to win, and you value keeping one code path for training and serving. It is nearly free to try — one line — and the [break-even accounting](/blog/machine-learning/performance-engineering/profiling-compiled-code) from the compiled-code post applies unchanged: a long-lived CPU service amortizes the C++ compile cost to nothing, a short batch job may not, so warm the cache. It is *not* the fastest option for a model that is dominated by large GEMMs on a well-supported CPU, where the BLAS library (oneDNN/MKL on Intel, its AMD equivalents) already runs near peak and there is nothing to fuse — there the compile win is single digits.

For those GEMM-dominated or latency-critical CPU services, the dedicated runtimes usually win: **ONNX Runtime** with its graph optimizations and oneDNN/OpenVINO execution providers, and **OpenVINO** on Intel hardware specifically, apply more aggressive operator fusion, layout optimization, and INT8 quantization than Inductor targets today, and they are built for exactly the CPU-serving case. The tradeoff is a heavier workflow — export to ONNX, a separate runtime, a second correctness gate — versus `torch.compile`'s one line. The cross-link for the export-and-runtime path is the [TensorRT and compiler](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler) post; the CPU story is analogous. And the moment your CPU service becomes truly compute-bound on big matmuls, the real question is whether it should be on a CPU at all — which loops back to the very first question this series asks about any service: where is the time actually going, and is this the right machine to spend it on?

The lever that most reliably separates a dedicated runtime from `torch.compile` on CPU is **INT8 quantization**. A CPU's integer throughput is far higher than its float throughput — Intel's VNNI instructions do a fused INT8 multiply-accumulate at four times the width of the float path, and the model also moves a quarter of the bytes through cache. For a GEMM-bound CPU service, quantizing to INT8 with ONNX Runtime or OpenVINO commonly buys a 2–4× latency reduction that no amount of float fusion in Inductor can match, because it changes the arithmetic the hardware runs, not just how the float loops are scheduled. The cost is an accuracy check (INT8 needs calibration and can shift outputs by a fraction of a percent) and the export pipeline. `torch.compile` does not target INT8 CPU codegen as a first-class path today, so once quantization is on the table, you are in the dedicated-runtime world. The decision table:

| Backend | Fusion | INT8 | GEMM speed | Workflow | Reach for it when |
|---|---|---|---|---|---|
| `torch.compile` Inductor | pointwise C++/SIMD | no (float focus) | oneDNN default | one line, stays in PyTorch | pointwise-heavy, staying in torch |
| ONNX Runtime | aggressive graph-level | yes, mature | oneDNN/tuned | export + runtime | GEMM-heavy, cross-platform |
| OpenVINO | aggressive + layout | yes, best on Intel | best on Intel | export + Intel runtime | Intel serving, latency-critical |

The honest reading of that table is that `torch.compile` is the right *default* — you try it first because it costs one line and it wins big on the pointwise-heavy models that make up a lot of real CPU inference — and you graduate to a dedicated runtime when the profile says you are GEMM-bound or when INT8's integer-throughput win is worth an export pipeline. The profile is what tells you which world you are in, which is the same discipline the whole series runs on.

## Case studies: real CPU compile numbers

A few results from the primary sources, framed honestly and with the caveat that exact numbers depend on the CPU, the compiler, the shapes, and the PyTorch version — treat them as order-of-magnitude, verify-on-your-own-hardware claims.

**Inductor CPU on pointwise-heavy models.** The PyTorch team's `torch.compile` CPU backend reports and the Inductor design notes describe the same mechanism this post centers on: the largest CPU speedups come from fusing chains of pointwise operators — normalizations, activations, residual adds, elementwise feature transforms — into single vectorized, OpenMP-threaded loops, with reported inference speedups clustering in the 1.3–2.0× range on models where that glue is a meaningful fraction of the work. Models dominated by large GEMMs see little, because oneDNN already runs the matmuls near peak and there is no pointwise chain to collapse — the "compile had nothing to do" null result the worked example above classifies correctly.

**oneDNN as the GEMM floor.** Intel's oneDNN (the library behind PyTorch's CPU `addmm`) is extremely well tuned for standard Transformer and CNN shapes on recent Xeons, which is *why* Inductor leaves GEMMs on it by default rather than generating its own matmul. The practical consequence, consistent across community benchmarks, is that the compile win on a CPU model is almost entirely a function of its non-GEMM fraction — a useful predictor before you even run the profile: the more normalization, activation, and elementwise work relative to matmul, the more `torch.compile` will help.

**INT8 on dedicated runtimes.** ONNX Runtime's and OpenVINO's quantization documentation and published benchmarks report 2–4× latency improvements from INT8 on GEMM-bound CPU inference on VNNI-capable Intel hardware, driven by the integer throughput and reduced memory traffic described above, at the cost of a calibration step and a small accuracy shift. This is the consistent, reproducible reason those runtimes beat float `torch.compile` on the compute-bound case — a different arithmetic, not a better schedule.

**The scalar-fallback tax.** The recurring, under-appreciated finding across CPU-compile debugging — and the reason this post insists on reading the generated `.cpp` — is that a silent scalar fallback (Inductor declining to vectorize a loop it cannot prove is safe, often a non-contiguous stride) is one of the most common causes of a compiled CPU kernel underperforming, and it is invisible in the profiler table, which shows only a slow kernel with no reason. The fix is a layout change upstream, and the diagnosis lives entirely in the generated code.

## Key takeaways

- **torch.compile works on CPU through the Inductor C++/OpenMP backend**, selected automatically for CPU tensors. It fuses pointwise chains, vectorizes with `at::vec` SIMD (AVX2/AVX-512/NEON), and threads with OpenMP — and you can read the generated `.cpp` to confirm all three.
- **The generated code is the ground truth.** `TORCH_LOGS="output_code"` prints the C++; a fused, vectorized, threaded loop has two `loadu`s, one `store`, an `at::vec` stride of 8 or 16, and a `#pragma omp parallel for`. Missing any one is your bug.
- **Debug wrong answers with the minifier.** `TORCHINDUCTOR_REPRO_AFTER="aot"` bisects a miscompiling graph to a minimal reproducer — the automated version of comparing against eager and shrinking the difference.
- **Profile to split host-bound from compute-bound.** `cpp_fused_` kernels prove fusion; a table of tiny `aten::` ops means host-bound (compile helps); a fat kernel near peak means compute-bound (vectorization width and thread count are your levers, not more fusion).
- **A scalar fallback is the silent CPU killer.** If Inductor cannot vectorize a loop (often a non-contiguous stride), it emits a scalar version that is 8–16× slower and looks merely "slow" in the profiler. Read the loop; fix the layout upstream.
- **Watch thread oversubscription.** `torch.set_num_threads` times replicas must not exceed physical cores; pin with `numactl`. Compilation cannot rescue contention you create around it.
- **Know when to leave PyTorch.** For GEMM-dominated or latency-critical CPU serving, ONNX Runtime and OpenVINO often beat Inductor with heavier fusion and INT8 — at the cost of an export step and a second runtime.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes and the profile→fix→measure loop, which apply to CPU services unchanged.
- [What torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) — Dynamo, guards, and Inductor; the front of the stack is identical on CPU, only the backend generator differs.
- [Profiling compiled code](/blog/machine-learning/performance-engineering/profiling-compiled-code) — reading fused kernel names and the compile-time break-even, which transfer directly to `cpp_fused_` kernels.
- [CPU affinity, NUMA, and threading](/blog/machine-learning/performance-engineering/cpu-affinity-numa-and-threading) — the thread-count and pinning discipline that keeps OpenMP from oversubscribing your cores.
- [When the CPU is your GPU bottleneck](/blog/machine-learning/performance-engineering/when-the-cpu-is-your-gpu-bottleneck) — the host-bound signature, which is the same disease Inductor's dispatch-elimination cures.
- [TensorRT: end-to-end inference compiler](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler) — the export-and-dedicated-runtime path that ONNX Runtime and OpenVINO take on CPU.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree tying every tool and fix together.
