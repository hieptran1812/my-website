---
title: "TensorRT, End to End: How the Inference Compiler Actually Works"
date: "2026-05-18"
publishDate: "2026-05-18"
description: "A deep dive into TensorRT as a compiler — graph import, layer and tensor fusion, kernel autotuning, precision calibration, the engine plan, dynamic shapes, and the full TensorRT-LLM path through in-flight batching, paged KV cache, FP8 attention, and multi-GPU sharding."
tags:
  [
    "tensorrt",
    "tensorrt-llm",
    "gpu",
    "inference",
    "mlops",
    "quantization",
    "cuda",
    "kernel-fusion",
    "model-serving",
    "nvidia",
    "fp8",
    "kv-cache",
  ]
category: "mlops"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

Most teams treat TensorRT as a black box with a magic speedup knob: feed it an ONNX file, get back something faster, ship it. That works right up until the engine fails to load on a different GPU, or INT8 quietly drops six points of accuracy, or a "30% faster" benchmark turns into a 10% regression in production because the batch size you tuned for is not the batch size you serve. Every one of those failures comes from the same root misunderstanding — TensorRT is not a runtime that runs your model. **TensorRT is a compiler that builds a new program, specialized to one GPU, one precision policy, and one shape regime, and then runs that.** Once you internalize that, the failure modes stop being mysterious and start being predictable.

This article is the end-to-end tour I wish I had when I first inherited a TensorRT serving stack: how the builder actually transforms your graph, why engines are not portable, what calibration is really doing to your activations, how the engine plan lays out memory, and then the entire TensorRT-LLM path — in-flight batching, paged KV cache, FP8 attention kernels, and tensor parallelism. We will quantify every claim and close with twelve production incidents and the reasoning a senior engineer should bring to each.

## Why TensorRT is a compiler, not a runtime

The single most expensive mental bug in this space is treating the build phase as setup and the run phase as the real work. They are two different programs with two different cost models, and conflating them produces wrong decisions at every level — CI design, deployment topology, accuracy validation, capacity planning.

| What people assume | The naive view | The reality |
|---|---|---|
| "TensorRT runs my PyTorch/ONNX model faster" | A runtime interprets the same graph more efficiently | The builder *compiles a new graph* — fused, retyped, kernel-specialized — and discards the original structure |
| "An engine is portable like a model file" | `.plan` is just weights plus topology | The `.plan` embeds GPU-specific kernel binaries; it is illegal to load on a different architecture |
| "Build is a one-time setup step" | Build cost does not matter | Build can take 5–60 minutes and dominates CI; it must be cached and version-pinned |
| "INT8 is a flag" | Set a flag, get 4× throughput, same accuracy | INT8 requires a calibration dataset, per-tensor scales, and accuracy re-validation; skipping that is how you ship a broken model |
| "Faster on batch 1 means faster everywhere" | Latency scales smoothly with batch | Kernels are autotuned for *one* shape; off-profile shapes can regress |

The compiler framing also tells you where to spend engineering effort. A compiler has a build phase you pay once and an execution phase you pay forever. You optimize them with completely different tools: the build phase wants caching, reproducibility, and version pinning; the execution phase wants profiling, memory accounting, and the right concurrency model. Mixing the two — for example, rebuilding an engine on every container start because "it's just initialization" — is one of the most common and most expensive mistakes I see.

## The mental model

![TensorRT build-once run-many compiler pipeline](/imgs/blogs/tensorrt-end-to-end-inference-compiler-1.png)

The diagram above is the mental model: a one-time **build phase** on the left, a repeated **execution phase** on the right, and a single frozen artifact — the engine plan — joining them. Everything in the build phase is expensive, non-deterministic in timing, and GPU-specific: layer and tensor fusion rewrite the graph, kernel autotuning physically benchmarks candidate kernels on the target device, and precision calibration measures activation ranges to pick quantization scales. The output is a serialized `.plan` file. The execution phase is cheap and repeatable: you deserialize the plan into an engine, create one or more execution contexts (each owning its own activation memory), and replay inference thousands of times.

The rest of this article is a guided tour of that diagram. We start at the network definition, walk through each builder transformation, examine the engine plan and its memory model, handle dynamic shapes, and then follow the same pipeline into TensorRT-LLM where the LLM-specific machinery — continuous batching, paged KV cache, FP8 — lives. Keep one rule in mind throughout: **anything that happens in the build phase is baked in; you cannot change it at runtime without rebuilding.**

If you have read the companion post on [CUDA Graphs](/blog/machine-learning/deep-learning/cuda-graph), the relationship is worth stating early: CUDA Graphs amortize *launch overhead* by replaying a captured stream of kernels; TensorRT chooses and fuses *the kernels themselves*. They compose — TensorRT-LLM uses CUDA Graphs internally to replay its decode step — but they solve different problems. TensorRT decides *what* to run; CUDA Graphs decide *how cheaply to submit* it.

## 1. The network definition layer

**Senior rule of thumb: TensorRT can only optimize what it can see — the graph it imports is the graph it compiles, so import fidelity is the ceiling on every speedup downstream.**

TensorRT does not consume PyTorch. It consumes a network definition built through one of three front doors: the ONNX parser (`trtexec --onnx=...` or `nvonnxparser` in C++/Python), the TensorRT network definition API directly (you call `addConvolution`, `addElementWise`, etc.), or a framework integration like Torch-TensorRT that traces a model and lowers supported subgraphs. In practice 90% of production use is the ONNX path, so that is what we will focus on.

The parser walks the ONNX graph and constructs an internal `INetworkDefinition` — a directed acyclic graph of `ILayer` nodes connected by `ITensor` edges. This IR is what the builder optimizes. Two consequences fall out immediately. First, any operator the parser does not recognize becomes a hard failure (or, with a plugin, an opaque box the builder cannot fuse across). Second, the *structure* of your exported ONNX graph matters: a model exported with a giant fused custom op gives the builder nothing to work with, while a model exported as clean primitive ops gives it maximum fusion surface.

Here is a realistic ONNX-to-engine build in the Python API. Note the explicit version checks and the things people forget — workspace memory pool limits, timing cache, and the precision flags:

```python
import tensorrt as trt

## TensorRT 10.x API. Pin this — the API churns between major versions.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str, engine_path: str,
                 fp16: bool = True, max_workspace_gb: int = 4) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))      # do not swallow these
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # Workspace is the tactic scratch budget — too small and the builder
    # silently skips the fast kernels that need more scratch.
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, max_workspace_gb << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Timing cache: reuse kernel timings across builds of similar graphs.
    # This is the single biggest build-time win in CI.
    cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

build_engine("model.onnx", "model.plan", fp16=True)
```

The thing to internalize: `build_serialized_network` is where minutes of wall time go. Everything before it is cheap parsing; everything inside it is the compiler doing fusion, autotuning, and calibration.

### Second-order optimization: export quality decides fusion quality

A non-obvious gotcha lives upstream of TensorRT entirely. If you export ONNX with `opset` too low, common patterns (LayerNorm, GELU, scaled-dot-product attention) come out as long chains of primitive ops, and while TensorRT *can* fuse some of them, it cannot fuse all of them. Export with a recent opset (17+) and the framework emits fused ops (`LayerNormalization`, `Attention`) that TensorRT maps directly to optimized kernels. I have seen a 1.4× difference on the *same model* purely from re-exporting with a higher opset. The lesson: profile the ONNX graph before you blame the builder.

## 2. Layer and tensor fusion

**Senior rule of thumb: fusion is the single largest source of TensorRT speedup on convolutional and small-tensor workloads, because it converts a memory-bandwidth-bound sequence of kernels into one compute-bound kernel.**

Modern GPUs are bandwidth-starved. An H100 does roughly 1,000 TFLOP/s of FP16 math but only ~3.35 TB/s of HBM bandwidth. For a small operation like a bias-add or a ReLU, the arithmetic is trivial and the entire cost is reading the tensor from HBM and writing it back. Run conv, bias, and ReLU as three separate kernels and you pay three full round-trips to HBM for what is, mathematically, one fused operation.

![Vertical fusion collapses a conv-bias-ReLU chain](/imgs/blogs/tensorrt-end-to-end-inference-compiler-2.png)

Vertical fusion — shown above — collapses a chain of layers along the data-flow direction into a single kernel. The classic case is the CBR pattern (Convolution + Bias + ReLU). Unfused, the conv kernel writes its feature map to global memory, the bias kernel reads and rewrites it, and the ReLU kernel does a third round-trip. Fused, the conv kernel computes a tile, adds the bias and applies the activation while the result is still in registers or shared memory, and writes the final tensor to HBM exactly once. The arithmetic is identical; the memory traffic drops by roughly 3×, and on a bandwidth-bound layer that is close to a 3× speedup.

Horizontal fusion is the orthogonal trick. When several layers consume the *same input tensor* — think the three projection convolutions feeding a multi-branch block, or the Q/K/V projections in attention — TensorRT can merge them into one wider kernel.

![Horizontal fusion merges sibling layers](/imgs/blogs/tensorrt-end-to-end-inference-compiler-3.png)

As the figure shows, the shared input is read once, the launch count drops from three to one, and the merged kernel covers the SMs at higher occupancy. On small batch sizes, where each individual kernel underfills the GPU, horizontal fusion is often worth more than vertical fusion: it is the difference between three kernels each using 30% of the SMs and one kernel using 90%.

The fusion catalog is large and version-dependent, but the categories are stable:

| Fusion type | Pattern | Why it wins |
|---|---|---|
| Conv-bias-activation (CBR) | `Conv → Add → ReLU` | Eliminates 2 HBM round-trips |
| Conv-conv (residual) | `Conv → ... → Add(skip)` | Fuses the residual add into the second conv's epilogue |
| Elementwise chains | `Mul → Add → Sigmoid` | Many tiny kernels become one |
| GEMM + epilogue | `MatMul → BiasAdd → GELU` | Activation runs in the GEMM epilogue, no extra kernel |
| Attention fusion | `MatMul → Softmax → MatMul` | Mapped to a flash-attention-style fused kernel |
| Constant folding | weights pre-computed at build | Removes ops entirely |

To *see* what fusion did, dump the engine layer information:

```bash
trtexec --onnx=model.onnx --fp16 \
        --exportLayerInfo=layers.json \
        --profilingVerbosity=detailed \
        --saveEngine=model.plan
```

The `layers.json` will show names like `Conv_0 + Relu_1` or `{ForeignNode[...]}` — those `+` signs are the fusions. Counting the layers before and after is the fastest sanity check that fusion happened at all.

Two things are worth knowing about the *limits* of fusion. First, fusion cannot cross an operation the builder does not understand — drop an unsupported op (handled by a plugin) into the middle of a CBR-friendly chain and you get two short fused regions on either side of an opaque box, not one long fused kernel. The plugin is a fusion barrier. Second, fusion is shape- and precision-dependent: a chain that fuses cleanly in FP16 may not fuse the same way in INT8 because the reformatting (cast) nodes the quantizer inserts sit between the layers you wanted merged. This is one more reason to inspect `layers.json` after every meaningful change rather than assuming last build's fusion still holds.

There is also a measurable scaling story behind fusion. On a small batch — batch 1, the latency case — a network spends most of its wall time in kernel launch overhead and bandwidth, and fusion's launch-count reduction is worth the most there; it is not unusual to see a 2× wall-time improvement on batch 1 from fusion alone. On a large batch, where each kernel is already compute-bound and the GPU is saturated, fusion's *relative* win shrinks because the bandwidth round-trips it eliminated were a smaller fraction of the total. The practical reading: if you benchmark fusion's benefit at large batch and conclude "fusion does not matter much," you have measured the regime where it matters least. Measure at the batch size you serve.

### Second-order optimization: fusion can hide your bugs

Here is the gotcha nobody warns you about. Once `Conv → Bias → ReLU` is fused, the intermediate tensors *no longer exist* — there is no buffer holding the post-conv, pre-ReLU activation, because it lived only in registers. If you are debugging a NaN or an accuracy regression and you want to inspect that intermediate, you cannot. You have to rebuild with fusion disabled at that layer (`config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)` plus per-layer precision, or in newer versions the debug-tensor API and `--markDebug`). The mental model: fusion is a compiler optimization, and like all compiler optimizations it makes the running program faster and the debugging experience worse. Budget for that.

## 3. Kernel autotuning (tactic selection)

**Senior rule of thumb: TensorRT does not *know* the fastest kernel for your layer — it *measures* it, on your GPU, at build time, which is exactly why an engine is not portable.**

This is the part of TensorRT that surprises people the most. For a single logical operation — say a 3×3 convolution at a given input shape — there are many ways to implement it on a GPU: an implicit GEMM, a Winograd transform, an FFT-based convolution, a CUTLASS template instantiation, a cuDNN call, each with multiple tile sizes and data layouts. NVIDIA does not ship a lookup table of "best kernel for shape X." Instead, the builder enumerates the candidate implementations — TensorRT calls them **tactics** — and physically runs each one on the target GPU, times it, and keeps the fastest measured one.

![Tactic selection times candidate kernels](/imgs/blogs/tensorrt-end-to-end-inference-compiler-4.png)

The figure shows the loop for one layer: four candidate tactics, four measured timings, the fastest one (CUTLASS at 0.29 ms) baked into the engine plan. Multiply that by every layer in the network and you understand both why builds take minutes and why the result is GPU-specific. The winning tactic for a 3×3 conv on an A100 — with its particular SM count, L2 size, and tensor-core generation — is frequently *not* the winning tactic on an H100 or an L4. The builder benchmarked on the build GPU; the timings are only valid for that architecture.

This produces the single hardest operational rule in TensorRT:

> An engine plan is married to the GPU architecture it was built on. Build on the architecture you serve on, or do not build at all.

Concretely: a `.plan` built on an A100 (compute capability 8.0) will refuse to deserialize on an L4 (8.9) or an H100 (9.0). It is not a soft fallback or a slow path — it is a hard `deserialize` failure. The corollary for CI: your build runners must have the same GPU SKU as production, or you build on production-class hardware and ship the artifact.

The build-time cost is real and tunable. The two levers that matter most:

| Lever | Effect | When to use |
|---|---|---|
| Timing cache | Reuses tactic timings across builds; can cut rebuild time 5–10× | Always, in CI |
| `--builderOptimizationLevel` (0–5) | Lower = fewer tactics tried, faster build, possibly slower engine | Level 3 for iteration, 5 for the shipped artifact |
| `DLA` / hardware-compatibility flags | Restricts tactic set | Only when targeting DLA or cross-arch |

```bash
## First build: pay the full autotuning cost, save the timing cache.
trtexec --onnx=model.onnx --fp16 \
        --builderOptimizationLevel=5 \
        --timingCacheFile=timing.cache \
        --saveEngine=model.plan

## Later builds of a similar graph reuse timing.cache and skip
## re-benchmarking tactics whose layer signature is unchanged.
```

It helps to be concrete about what a "tactic" actually is, because the word sounds abstract. A tactic is a specific compiled kernel implementation with a specific set of compile-time parameters: which algorithm family (implicit GEMM, Winograd, FFT, direct), which tile size, which data layout (NCHW, NHWC, or a tensor-core-friendly interleaved layout), which split-K factor, whether it uses the tensor cores or the CUDA cores. TensorRT's kernel libraries — cuDNN, cuBLAS, cuteGEMM/CUTLASS-derived kernels, and TensorRT's own hand-written kernels — collectively expose thousands of these. For one convolution at one shape, dozens may be applicable. The builder does not reason about them analytically; it runs each applicable tactic a few times, takes the best observed time, and moves on. That is the whole algorithm, and its simplicity is exactly why it is robust across hardware generations NVIDIA had not designed when the heuristic was written — a measurement does not go stale the way a hand-tuned cost model does.

This also explains the shape of the build-time/quality tradeoff. `builderOptimizationLevel` does not change *what* the engine computes; it changes how many tactics the builder is willing to try. At level 0 it tries a small set and trusts heuristics; at level 5 it tries the widest set and measures exhaustively. The engine from level 5 is usually a few percent faster than level 3 and meaningfully faster than level 0 — but the build can take several times longer. The right policy follows directly from the compiler framing: iterate at a low level where build speed matters, and ship the artifact built at level 5 where the few percent runtime gain is paid back over every inference for the life of the deployment.

### Second-order optimization: the timing cache is shape- and version-sensitive

The timing cache keys on a hash of the layer's signature — op type, shapes, precision, GPU. Change the GPU and the cache entries miss (correctly). Upgrade TensorRT and the cache may invalidate wholesale because tactic IDs changed. The trap: teams check a `timing.cache` into the repo, upgrade TensorRT, and then silently get *worse* engines because they set `ignore_mismatch=True` and the builder reused stale or partial timings. Pin the timing cache to the TensorRT version, and let mismatches fail loudly.

## 4. Precision: FP16, INT8, FP8

**Senior rule of thumb: lower precision buys you bandwidth and tensor-core throughput, but every step down the precision ladder is an accuracy contract you must explicitly validate — TensorRT will happily build a fast, wrong engine.**

Precision is where TensorRT delivers its biggest *and* most dangerous wins. The mechanics are simple to state and subtle to get right.

**FP16** is almost free. Weights and activations are 16-bit; tensor cores run FP16 matmuls at 2× the FP32 rate; memory traffic halves. For the vast majority of CNNs and transformers, FP16 inference is accuracy-neutral — the dynamic range of `half` is wide enough. FP16 should be your default; you turn it on with one flag and re-check your eval metric once.

**INT8** is where it gets interesting. Integers have no exponent — an 8-bit signed integer represents 256 evenly spaced values. To map a floating-point tensor onto that grid you need a **scale**: `quantized = round(real / scale)`, `real ≈ quantized × scale`. Pick the scale too large and you waste resolution on values that never occur; pick it too small and large activations *clip*, destroying information. The scale is not something you can guess — it depends on the actual distribution of activations flowing through that specific tensor, which depends on your data.

That is what **calibration** solves.

![INT8 calibration picks a dynamic range](/imgs/blogs/tensorrt-end-to-end-inference-compiler-5.png)

The calibration pipeline above is post-training quantization (PTQ): you feed TensorRT a few hundred representative inputs, it runs the FP32 network and records a histogram of every tensor's activation magnitudes, then for each tensor it sweeps candidate clipping thresholds and picks the one that minimizes the KL divergence between the original FP32 distribution and the distribution you would get after quantizing at that threshold. The output is one scale factor per tensor (or per channel for weights), baked into the engine.

You implement the calibrator by handing TensorRT batches of real data:

```python
import tensorrt as trt
import numpy as np

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """Entropy calibration = KL-divergence threshold search."""
    def __init__(self, data_batches, cache_file="calib.cache"):
        super().__init__()
        self.batches = iter(data_batches)   # iterable of np.ndarray
        self.cache_file = cache_file
        self.device_input = None            # allocate a CUDA buffer once

    def get_batch_size(self):
        return 8

    def get_batch(self, names):
        try:
            batch = next(self.batches)      # one calibration batch
        except StopIteration:
            return None                     # signals "calibration done"
        # copy batch to self.device_input (cuda memcpy) and return ptr
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # If we calibrated before, reuse the scales — skip the data pass.
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

## Wire it into the build config:
## config.set_flag(trt.BuilderFlag.INT8)
## config.int8_calibrator = Int8Calibrator(my_batches)
```

Two details decide whether INT8 ships or embarrasses you. First, **the calibration set must be representative** — same preprocessing, same domain, same distribution as production traffic. Calibrate a detector on daytime images and serve it at night and your scales clip the wrong tail. A few hundred diverse samples beat ten thousand near-duplicates. Second, **per-channel weight quantization** matters: a convolution's output channels can have wildly different weight magnitudes, and a single per-tensor weight scale forces them to share resolution. Per-channel scales (the default for weights in modern TensorRT) routinely recover several points of accuracy.

When PTQ is not enough, the answer is **quantization-aware training (QAT)**: you insert fake-quant nodes during fine-tuning so the network *learns* weights that survive INT8, then export with the scales already embedded (TensorRT reads them from `QuantizeLinear`/`DequantizeLinear` ONNX nodes and skips calibration). QAT costs a training run but routinely closes the last 1–2 points PTQ leaves on the table.

**FP8** (E4M3 on Hopper and newer) is the LLM-era precision. Unlike INT8 it *has* an exponent, so it tolerates the heavy-tailed activation distributions of transformers far better, and it needs much lighter calibration — often just a per-tensor amax scale. FP8 matmuls run at the same blistering tensor-core rate as INT8 but with INT8-grade memory savings and FP16-grade robustness. The catch: FP8 tensor cores exist only on compute capability 8.9+ (Ada/Hopper). Build an FP8 engine and run it on Ampere and it will either fail or silently fall back to a slow emulated path.

| Precision | Bits | Tensor-core speedup vs FP32 | Calibration needed | Typical accuracy hit | HW requirement |
|---|---|---|---|---|---|
| FP32 | 32 | 1× | none | baseline | any |
| TF32 | 19 (10 mantissa) | ~4× | none | negligible | Ampere+ |
| FP16 | 16 | ~8× | none | ~0 | Volta+ |
| INT8 | 8 | ~16× | yes (PTQ/QAT) | 0.5–3 pts if done right | Turing+ DP4A |
| FP8 (E4M3) | 8 | ~16× | light (amax) | <1 pt typical | Ada/Hopper+ |

The companion post on [quantization tradeoffs at the edge](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) goes deeper on the numeric formats themselves; here the point is operational — **precision is a build-phase decision with a runtime accuracy consequence, and the only safe workflow is to re-run your evaluation metric on the built engine, never on the source model.**

#### Why FP8 changed the LLM calculus

It is worth dwelling on why FP8 specifically reset the precision conversation for language models. INT8's failure mode on transformers is *outliers*: a handful of activation channels in an LLM carry magnitudes one or two orders larger than the rest, and because INT8 has no exponent, a per-tensor scale wide enough to represent those outliers leaves almost no resolution for the bulk of the distribution. The community spent real effort working around this — outlier-aware schemes, per-channel activation quantization, mixed-precision decompositions that keep the outlier channels in FP16. FP8's E4M3 format sidesteps the whole problem: with four exponent bits it represents a wide dynamic range natively, so the heavy tail is not a catastrophe, and an amax-based per-tensor scale is usually enough. That is why TensorRT-LLM treats FP8 as the default high-performance precision on Hopper and INT8 as the more delicate option — the opposite of the CV world, where INT8 is still the workhorse because CNN activations are far better behaved. The general lesson: the right low precision depends on the *shape of your activation distribution*, and transformers and CNNs do not have the same shape.

### Second-order optimization: mixed precision and the precision constraint trap

TensorRT picks per-layer precision to maximize speed, which means it may run a numerically sensitive layer (a softmax, a normalization, a final logits projection) in INT8 even when you would rather it not. If you observe an accuracy cliff localized to one part of the network, you can pin specific layers to higher precision with `layer.precision = trt.float16` plus the `OBEY_PRECISION_CONSTRAINTS` flag. The gotcha: every precision pin you add forces a reformatting kernel (a cast) at the boundary, and those casts cost bandwidth. Pin the three layers that matter, not thirty.

## 5. The engine plan and the memory model

**Senior rule of thumb: an engine reserves three distinct kinds of memory — weights, activation scratch, and build-time workspace — and confusing them is how you both over-provision your fleet and get surprised by OOM under concurrency.**

When you deserialize a `.plan`, the result is an `ICudaEngine`. The engine holds the weights and the kernel binaries — that memory is fixed and shared. To actually run inference you create an `IExecutionContext`, and *the context* owns the activation memory: the scratch buffers that intermediate tensors flow through.

![What the engine plan reserves at runtime](/imgs/blogs/tensorrt-end-to-end-inference-compiler-6.png)

The figure shows the three regions. **Weights** are fixed — serialized into the plan, identical no matter how many contexts you create. **Activation memory** is reused scratch: TensorRT computes the lifetime of every intermediate tensor at build time and, because most tensors are dead long before inference ends, it overlays non-overlapping lifetimes onto the same physical bytes. A 100-layer network does not need 100 activation buffers live at once; it needs as many as the widest point of the live-range graph. This is why TensorRT's activation footprint is often dramatically smaller than a naive "sum of all tensors" estimate. **Workspace** is a build-time concept — the scratch budget you grant the builder so it can consider tactics that need temporary memory.

The operational consequence is the part teams get wrong. The shared weights cost you once. The activation memory costs you **per execution context**. If you want to serve N concurrent requests through N contexts on one GPU, your memory bill is `weights + N × activation_size`, not `N × (weights + activation)`. Get this right and one GPU serves far more concurrency than you expected; get it wrong and you either waste VRAM or OOM at the worst possible moment.

```python
import tensorrt as trt

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open("model.plan", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

## Weights are loaded once, here. Cheap to query the footprint:
print(f"device memory / context: {engine.device_memory_size >> 20} MB")

## Each context = one more activation arena. Create one per concurrent
## in-flight request; they share the engine's weights.
ctx_a = engine.create_execution_context()
ctx_b = engine.create_execution_context()
## Total VRAM ≈ weights + 2 × engine.device_memory_size
```

Serialization is the other half of the story. `build_serialized_network` returns bytes; you write them to disk and that file *is* your deployable artifact. It contains kernel binaries, weights, fusion structure, and tactic choices. It does **not** contain a guarantee of forward compatibility — which is the subject of the next gotcha.

### Second-order optimization: lazy module loading and weight streaming

Two newer features change the memory math. CUDA lazy module loading (`CUDA_MODULE_LOADING=LAZY`, the default on recent drivers) means kernel binaries page in on first use rather than all at once, cutting deserialize time and idle VRAM. And TensorRT 10's **weight streaming** lets an engine keep weights in host memory and stream them to the GPU on demand — trading bandwidth for VRAM, which is the only way to fit a model whose weights exceed device memory. Both are build- or runtime-config decisions; neither is automatic in the way people assume.

## 6. Dynamic shapes and optimization profiles

**Senior rule of thumb: a static-shape engine is a benchmark; a production engine almost always needs dynamic shapes — and the optimization profile you choose silently decides which shapes are fast and which are merely correct.**

Real serving traffic does not arrive at one fixed shape. Batch size varies with load; sequence length varies with input; image resolution varies with the client. TensorRT handles this with **dynamic shapes**: you mark certain tensor dimensions as `-1` (runtime-determined) and supply an **optimization profile** that bounds them.

![Optimization profiles tune for the opt shape](/imgs/blogs/tensorrt-end-to-end-inference-compiler-7.png)

A profile is three shapes per dynamic input: `min`, `opt`, and `max`. As the figure makes explicit, the builder autotunes kernels for the **opt** shape. The min and max bounds tell the builder the legal range so it can allocate enough activation memory and reject illegal shapes — but the tactic selection, the thing that actually determines speed, is optimized for `opt` alone. Run at `min` or `max` and the engine is *correct* but uses kernels chosen for a different shape; it can be meaningfully slower.

This is the source of the "benchmark lied" failure. A team benchmarks at batch 1 (because that is the easy latency number), sets `opt` to batch 1 by default, then serves production at batch 16–32. Every kernel was tuned for batch 1; the batched path runs on suboptimal tactics; throughput is well below what a batch-32-tuned engine would deliver. **Set `opt` to the shape you actually serve most**, not the shape that is convenient to benchmark.

```bash
trtexec --onnx=bert.onnx --fp16 \
  --minShapes=input_ids:1x16,attention_mask:1x16 \
  --optShapes=input_ids:16x256,attention_mask:16x256 \
  --maxShapes=input_ids:32x512,attention_mask:32x512 \
  --saveEngine=bert.plan
```

You can attach **multiple optimization profiles** to one engine — say one tuned for short sequences and one for long — and select the profile per execution context at runtime. Each profile adds activation memory (the engine sizes for the largest), so there is a VRAM cost, but for bimodal traffic it is the clean answer.

| Approach | Latency at opt shape | Latency off-shape | VRAM | When to use |
|---|---|---|---|---|
| Static engine per shape | Best | N/A (wrong shape = rebuild) | Lowest | Fixed-shape pipelines |
| One dynamic profile | Best at opt | Degraded | Moderate | Unimodal traffic |
| Multiple profiles | Best at each opt | Degraded between | Highest | Bimodal/multimodal traffic |

### Second-order optimization: shape changes trigger context re-planning

Setting an input shape on an execution context is not free. When you call `set_input_shape` with a shape the context has not seen, TensorRT runs a shape-resolution pass that recomputes tensor sizes and memory offsets. If your serving loop changes shape on *every* request, you pay that pass every request. The fix is to bucket: quantize incoming shapes to a small set (e.g., pad sequence length up to the next multiple of 128) so the context sees a handful of distinct shapes and the shape-resolution result stays warm. This is exactly the bucketing pattern good LLM servers use, and it is the bridge to the TensorRT-LLM half of this article.

## 7. TensorRT-LLM: in-flight batching

**Senior rule of thumb: for autoregressive LLMs, the scheduler matters more than the kernels — static batching wastes most of your GPU on stragglers, and in-flight batching is the fix that everything else builds on.**

Everything so far applies to any model. But large language models break two assumptions that the core TensorRT design quietly relies on: their compute graph runs *many times per request* (once per generated token), and different requests in a batch finish at different, unpredictable times. TensorRT-LLM is the library that specializes the whole pipeline for that reality. It still produces an engine plan, still fuses and autotunes and quantizes — but it adds a runtime with an LLM-aware request scheduler.

The first and most important thing that scheduler does is **in-flight batching**, also called continuous batching.

![Static batching vs in-flight batching](/imgs/blogs/tensorrt-end-to-end-inference-compiler-8.png)

Consider static batching first, on the left of the figure. You collect 8 requests, run them as a batch, and decode token by token. But generation lengths differ — request 3 emits its end-of-sequence token after 40 tokens while request 7 runs to 400. Under static batching, request 3's slot in the batch sits *idle* for 360 steps because the batch cannot retire until the slowest member finishes. With realistic length distributions, static batching leaves 50–70% of the GPU's decode capacity unused.

In-flight batching, on the right, fixes this at the scheduler level. The scheduler operates per *iteration* (per token step), not per batch. After every decode step it checks: has any sequence finished? If so, evict it and admit a waiting request into the freed slot — even though the other sequences in the batch are mid-generation. The batch is no longer a fixed cohort that starts and ends together; it is a rolling set of slots, continuously refilled. The result is 2–4× higher throughput at the same per-token latency, which is why every serious LLM serving stack — TensorRT-LLM, vLLM, [SGLang](/blog/machine-learning/large-language-model/sglang-inference) — implements some version of it.

There is a subtlety that makes in-flight batching possible: **prefill and decode are different shapes**. Prefill processes the whole prompt (a long sequence, compute-bound, FLOPs-heavy); decode processes one token (sequence length 1, memory-bandwidth-bound, dominated by reading the KV cache and weights). The scheduler interleaves them — admitting a new request means running its prefill, then folding it into the ongoing decode batch. TensorRT-LLM exposes policies for how aggressively to interleave (`max_num_tokens`, chunked prefill) so a long prompt's prefill does not stall every other request's decode.

```python
from tensorrt_llm import LLM, SamplingParams

## TensorRT-LLM builds the engine, then serves it with the in-flight
## batching runtime. The scheduler config is the lever that matters.
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    # in-flight batching is on by default in the C++ runtime;
    # these knobs bound it:
    max_num_seqs=256,        # max concurrent slots
    max_num_tokens=8192,     # token budget per iteration (prefill+decode)
    enable_chunked_prefill=True,   # split long prefills across iterations
)

params = SamplingParams(temperature=0.7, max_tokens=512)
for out in llm.generate(prompts, params):
    print(out.outputs[0].text)
```

#### Speculative decoding rides on the same scheduler

The in-flight batching scheduler is also what makes speculative decoding practical inside TensorRT-LLM. The idea: decode is memory-bandwidth-bound, so the GPU has spare compute during each decode step; spend it by having a small draft model (or a set of lightweight heads, as in Medusa or EAGLE) propose several tokens at once, then verify all of them with a single forward pass of the large model. If the draft is right, you got several tokens for the price of one step; if it is wrong, you fall back. TensorRT-LLM implements this as a build-time and runtime feature, and the reason it composes cleanly is that the scheduler already thinks in iterations and variable-length steps — accepting three speculative tokens or one is just another shape the iteration loop handles. The throughput win depends entirely on the draft model's acceptance rate, which is workload-dependent: 2–3× on predictable text, far less on high-entropy output. The broader treatment is in the post on [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding); the point here is structural — speculative decoding is not a separate runtime, it is a policy layered on the same in-flight batching machinery, which is why a well-designed LLM scheduler is the foundation everything else stands on.

### Second-order optimization: prefill can starve decode

The non-obvious failure mode: a single 32k-token prompt arrives, its prefill is enormous, and if the scheduler runs that prefill as one monolithic step every other request's decode freezes for the duration — a latency spike visible as a p99 cliff. `enable_chunked_prefill` is the fix: the long prefill is sliced into chunks that interleave with decode steps, trading a small prefill-throughput loss for a large p99 win. Tuning `max_num_tokens` is how you balance the two. This is the LLM-serving analog of bucketing shapes in core TensorRT — covered in more depth in [optimizing LLM inference](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide).

## 8. TensorRT-LLM: paged KV cache and FP8 attention

**Senior rule of thumb: the KV cache, not the model weights, is what runs you out of memory under load — and contiguous KV allocation wastes so much of it that paging is not an optimization, it is a requirement.**

During decode, every attention layer needs the keys and values of all previous tokens. Recomputing them each step would be quadratic; instead they are cached — the KV cache. The cache is large: for a model with `L` layers, `H` KV heads, head dimension `d`, in FP16, each token costs `2 × L × H × d × 2` bytes. For an 8B-class model that is on the order of tens of kilobytes *per token*; a 4k-token sequence is hundreds of megabytes, and you are serving hundreds of sequences.

The naive implementation reserves a contiguous buffer per sequence sized for the *maximum* possible length. That is catastrophic: a request that generates 50 tokens but was allocated for 4,096 wastes 98% of its reservation, and because the buffers are contiguous you cannot reclaim the gaps. Under in-flight batching — where sequences come and go constantly — contiguous allocation fragments the cache into uselessness within minutes.

![Paged KV cache indexes blocks like virtual memory](/imgs/blogs/tensorrt-end-to-end-inference-compiler-9.png)

The paged KV cache, shown above, borrows the idea straight from operating-system virtual memory. The cache is carved into fixed-size **blocks** — typically 16 or 32 tokens of KV each. A sequence does not own a contiguous region; it owns a **block table**, a small array mapping its logical token positions to physical block IDs. Blocks are allocated from a shared pool on demand as a sequence grows, and returned to the pool when it finishes. A 50-token sequence uses 4 blocks and frees them on completion; nothing is reserved for the 4,096-token length it never reached. The internal fragmentation drops from ~98% to at most one partly-filled block per sequence.

Paging unlocks a second win: **prefix sharing**. If 100 requests share the same system prompt, their identical prefix tokens can point at the *same physical blocks* — the KV for that prefix is computed once and shared, with copy-on-write only when the sequences diverge. For chat workloads with long shared system prompts this is a large, free memory and compute saving. The mechanics here are the same family as [KV cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) in the general LLM-serving literature.

On top of paging, TensorRT-LLM ships **fused attention kernels** — XQA and the FMHA family — that compute attention without ever materializing the full `[seq, seq]` score matrix in HBM, the same flash-attention principle of keeping the softmax online and tiled. And it can store the KV cache itself in **FP8**, halving the cache's memory footprint versus FP16. Since decode is bandwidth-bound on exactly that cache, an FP8 KV cache is close to a 2× decode throughput win — and because FP8 has an exponent, the accuracy cost on KV is small. The configuration:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, QuantConfig

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    quant_config=QuantConfig(
        quant_algo="FP8",          # FP8 weights + activations
        kv_cache_quant_algo="FP8", # FP8 KV cache — halves cache VRAM
    ),
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.9,   # how much VRAM the pool may claim
        enable_block_reuse=True,        # prefix sharing across requests
    ),
)
```

It is worth being precise about why the fused attention kernels matter as much as the paging. A naive attention implementation, for a sequence of length `n`, materializes an `n × n` score matrix in HBM, applies softmax over it, and multiplies it back against the values — that is `O(n²)` memory traffic for a result that is only `O(n × d)`. The XQA and FMHA kernels never write that matrix out: they tile the computation, keep a running softmax statistic in registers, and stream over the KV blocks, so the score matrix exists only in on-chip memory a tile at a time. This is the same online-softmax principle as FlashAttention. The interaction with paging is the elegant part — because the kernel already streams over KV in tiles, it can stream over *paged* KV blocks just as easily, walking the block table to find each block's physical address. Paging and fused attention are not two features bolted together; they are co-designed, which is why you should treat "enable paged context FMHA" as a single decision rather than two.

One more quantified point on the FP8 KV cache. Decode throughput is governed by how fast you can read the KV cache out of HBM for every token of every sequence in the batch. Halving the cache's bytes-per-token by storing it in FP8 instead of FP16 nearly halves that read traffic, and since decode is bandwidth-bound on exactly that read, the decode-side throughput improvement is close to linear in the saving. The accuracy cost is modest because the KV cache is a *storage* format, not an accumulation format — the attention math still accumulates in higher precision; only the stored keys and values are FP8. This is why an FP8 KV cache is one of the highest-return, lowest-risk knobs in the entire TensorRT-LLM configuration surface on Hopper-class hardware.

### Second-order optimization: the KV pool sizing tug-of-war

`free_gpu_memory_fraction` is a deceptively important knob. The KV block pool claims that fraction of *leftover* VRAM after weights and activations. Set it too low and you can serve only a few concurrent sequences before the pool is exhausted and the scheduler starts queuing — throughput collapses. Set it too high and a transient activation spike OOMs the process. The right value is empirical, found by loading the server to its target concurrency and watching the pool's high-water mark. Treat it as a capacity-planning parameter, not a default.

## 9. Multi-GPU: tensor and pipeline parallelism

**Senior rule of thumb: you go multi-GPU because the model does not fit or one GPU is too slow — and which parallelism you pick determines whether your bottleneck is interconnect bandwidth or pipeline bubbles.**

When a model exceeds one GPU's memory, or one GPU cannot hit your latency target, TensorRT-LLM shards it. There are two axes, and they are not interchangeable.

**Tensor parallelism (TP)** splits each layer *across* GPUs.

![Tensor parallelism shards one layer across GPUs](/imgs/blogs/tensorrt-end-to-end-inference-compiler-10.png)

As the figure shows, every weight matrix in a transformer layer is partitioned — attention heads are dealt out across GPUs, the MLP's columns are split — so each GPU holds a slice of every layer and computes a *partial* result. Because each GPU only computed part of the output, the partials must be summed: an **all-reduce** collective over NCCL, once per layer (twice, really — after attention and after the MLP). TP gives you low latency, because all GPUs work on the same token simultaneously, but it is communication-heavy: that per-layer all-reduce demands fast interconnect. On an NVLink-connected node TP scales well; across PCIe or across nodes the all-reduce becomes the bottleneck and TP efficiency falls off a cliff.

**Pipeline parallelism (PP)** splits the model *by depth* — GPU 0 holds layers 1–20, GPU 1 holds 21–40, and so on. Activations pass from one stage to the next, so the only communication is a point-to-point handoff at stage boundaries, far cheaper than an all-reduce. The cost is the **pipeline bubble**: while GPU 0 processes the first micro-batch, GPUs 1–3 sit idle waiting for work to reach them. PP tolerates slow interconnect (it works across nodes) but needs enough in-flight micro-batches to keep every stage busy.

| Strategy | Splits | Communication | Latency | Best when |
|---|---|---|---|---|
| Tensor parallel | Each layer's weights | All-reduce per layer (heavy) | Low | NVLink node, latency-critical |
| Pipeline parallel | Layers by depth | Point-to-point at boundaries (light) | Higher (bubble) | Cross-node, slow interconnect |
| TP + PP hybrid | Both | Both | Tuned | Very large models on multi-node |

```bash
## Build a 70B engine sharded 4-way with tensor parallelism.
trtllm-build --checkpoint_dir ./llama70b_fp8_ckpt \
             --output_dir ./llama70b_engine \
             --tp_size 4 --pp_size 1 \
             --max_batch_size 64 \
             --use_paged_context_fmha enable \
             --gemm_plugin fp8
```

The decision rule is mechanical. Within one NVLink-connected node, prefer TP — the all-reduce is cheap there and you get the latency win. Spanning nodes, you almost always need PP for the inter-node hop (the all-reduce over a slower fabric would dominate) and TP within each node. The companion post on [choosing a GPU for LLM serving](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) works the cost side of this; the point here is that the parallelism choice is baked into the engine at `trtllm-build` time — changing `tp_size` means rebuilding.

### Second-order optimization: TP degree changes your KV cache math

A subtle interaction: tensor parallelism shards the KV cache too, because each GPU only stores the KV for its slice of the attention heads. Going from TP=2 to TP=4 halves the per-GPU KV footprint, which means the KV pool can hold more concurrent sequences per GPU. So TP is not only a latency/throughput lever — it is also a concurrency lever, because it changes how much KV each GPU has to hold. People tune TP for latency and are then surprised that it moved their max-concurrency number; it is the same knob.

## 10. Serving the engine in production

**Senior rule of thumb: a fast engine wrapped in a slow serving loop is a slow service — the engine is the easy 80%, and the concurrency model around it is the hard 20% that decides your real throughput.**

A built `.plan` is inert. Something has to deserialize it, feed it inputs, and manage concurrency, and that wrapper is where a surprising amount of production performance is won or lost. There are three common ways to host a TensorRT engine, and the choice has real consequences.

![Three ways to host a TensorRT engine](/imgs/blogs/tensorrt-end-to-end-inference-compiler-11.png)

The figure above lays out the three hosting paths and the one concurrency rule that cuts across all of them: throughput scales with the number of execution contexts and CUDA streams you run in parallel, and sharing a single context behind a lock collapses that scaling to one-request-at-a-time regardless of how fast the engine is.

The **raw Python/C++ runtime** is the lowest-level option: you deserialize the engine yourself, manage CUDA streams, allocate input/output buffers, and call `execute_async_v3`. It is the right choice when TensorRT is one stage inside a larger custom pipeline — a preprocessing step, the engine, a postprocessing step — and you want full control over the stream graph. The trap is concurrency. A single execution context is **not** thread-safe for concurrent `execute` calls; to serve N requests in parallel you need N contexts (each with its own activation memory, as section 5 explained) and ideally N CUDA streams so the GPU's scheduler can overlap them. Teams that share one context across threads serialize all their inference behind a lock and then wonder why a "fast" engine yields low throughput.

**NVIDIA Triton Inference Server** is the batteries-included option. You drop the `.plan` into a model repository with a `config.pbtxt`, and Triton handles HTTP/gRPC endpoints, health checks, metrics, multi-model hosting, and — critically — **dynamic batching**: it holds incoming requests for a few milliseconds to assemble a larger batch before calling the engine. For a CV model served at moderate QPS, Triton's dynamic batcher is the single biggest throughput lever, because it turns a stream of batch-1 requests into the batch-16 shape your engine was autotuned for. The configuration is small:

```
## config.pbtxt for a TensorRT engine in Triton
platform: "tensorrt_plan"
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 3000
}
instance_group [ { count: 2, kind: KIND_GPU } ]
```

The `instance_group` count is the number of execution contexts Triton creates — the concurrency knob. `max_queue_delay_microseconds` is the latency you are willing to trade for batch assembly: too low and batches stay small, too high and p99 suffers. Tune it against your actual QPS.

The TensorRT-LLM runtime (the C++ `executor` API, or the Python `LLM` class, or Triton's TensorRT-LLM backend) is what you use for language models, because dynamic batching is the wrong model there — you need the *iteration-level* in-flight batching of section 7, not request-level batch assembly. Request-level batch assembly waits to gather whole requests before a call; iteration-level scheduling reshapes the batch every single token step, which is the only model that fits autoregressive generation. Do not try to serve an LLM engine through Triton's generic dynamic batcher; route it through the TensorRT-LLM backend so the LLM-aware scheduler is in the loop, and reach for the generic batcher only for fixed-graph models.

| Hosting option | Batching | Best for | Main pitfall |
|---|---|---|---|
| Raw Python/C++ runtime | You implement it | Engine embedded in a custom pipeline | Sharing one context across threads |
| Triton (`tensorrt_plan`) | Request-level dynamic batching | CV / fixed-graph models at scale | Mis-tuned queue delay |
| TensorRT-LLM runtime | Iteration-level in-flight batching | Autoregressive LLMs | Using the generic batcher instead |

### Second-order optimization: pin the preprocessing, not just the engine

A subtle production loss: teams optimize the engine to 2 ms and leave a 6 ms Python preprocessing step (image decode, resize, normalize) on the critical path, single-threaded, on the CPU. The engine is now 25% of the latency budget. The fixes are well-known but easy to forget — move preprocessing onto the GPU (NVIDIA DALI, or a small TensorRT engine for the resize/normalize), overlap it with inference on a separate CUDA stream, and batch it. Profile the *whole* request, not just `execute_async`. An engine that is fast in a microbenchmark and slow in production almost always has its time hiding in the glue code around it.

## Cross-cutting concerns

Three concerns cut across every layer above and deserve their own treatment.

### Observability: profile the engine, not the model

Once TensorRT has fused and retyped your graph, your framework-level profiler is blind — the PyTorch op names are gone. You profile the *engine*. `trtexec --dumpProfile` gives per-layer timing on the fused graph; for the real picture, capture an Nsight Systems trace (`nsys profile`) and look at the GPU timeline. The two things to check first: are there gaps between kernels (launch-bound — reach for CUDA Graphs), and is any single fused layer dominating (a tactic that autotuning got wrong, or a layer stuck in a high precision). TensorRT-LLM additionally exposes iteration-level stats — tokens/s, active request count, KV pool utilization — and those are the metrics that belong on your serving dashboard, not GPU-percent.

### Versioning and portability: the artifact is fragile

Restate the rule because it causes the most production incidents: an engine plan is specific to **(GPU architecture, TensorRT version, and to a lesser degree CUDA/driver version)**. TensorRT offers a hardware-compatibility mode and a version-compatible mode that relax the GPU and version coupling respectively — but both cost performance (they restrict the tactic set), so they are an escape hatch, not a default. The disciplined workflow: pin the TensorRT version in your container, build on the production GPU SKU, store the `.plan` in an artifact registry keyed by `(model, trt_version, gpu_arch)`, and treat a TensorRT upgrade as a full rebuild-and-revalidate of every engine. Never build an engine inside the serving container at startup against whatever driver happens to be there.

### Build cost: treat it like a compile

A cold TensorRT-LLM build of a large model is a 10–40 minute job. That is a compile, and it belongs in CI, cached, not on the critical path of a deploy or — worse — a pod restart. Use the timing cache, pin `builderOptimizationLevel`, build once per `(model, precision, shape-profile, gpu)` tuple, and ship the plan as an immutable artifact. A team that rebuilds engines on autoscale events has built a system whose scale-up latency is measured in tens of minutes; that is an outage waiting for a traffic spike.

## Case studies from production

### 1. The portable engine that wasn't

A team built TensorRT engines in a CI job running on the cheap GPU their CI provider offered — a T4 — and deployed to an A100 inference fleet. Local smoke tests passed because the smoke-test runner was also a T4. In production the service crash-looped: `deserialize_cuda_engine` returned null with a terse log line about an incompatible engine. The wrong first hypothesis was a corrupt artifact or a bad download. The actual root cause: the `.plan` embedded T4 (compute capability 7.5) kernel binaries, illegal to load on an A100 (8.0). The fix was to move the build job onto an A100 runner and key the artifact cache by GPU architecture so a T4-built plan could never be deployed to A100 hardware. The lesson: an engine plan is not a model file; it is a compiled binary, and you would never ship an x86 binary to an ARM server.

### 2. The INT8 accuracy cliff

An object detector that scored 38.5 mAP in FP32 dropped to 32.1 mAP after a one-line "enable INT8" change. The team's first instinct was that INT8 was simply too lossy for detection and they should abandon it. Wrong. Two mistakes compounded. First, the calibration set was 64 images pulled from a single scene — not representative of the deployment distribution — so the activation histograms were skewed and the scales clipped real traffic. Second, they were on an old enough TensorRT path that weight quantization defaulted to per-tensor. Switching to a 500-image calibration set sampled across the full deployment distribution recovered most of it; per-channel weight quantization recovered the rest, landing at 38.0 mAP — a 0.5-point loss for nearly 4× throughput. The lesson: an INT8 accuracy cliff is almost always a calibration-data or scale-granularity bug, not a fundamental limit.

### 3. The 40-minute build

A team's CI pipeline ballooned to 40 minutes, almost all of it in the TensorRT build step, and every model iteration paid it. The wrong hypothesis was that the model had simply gotten too big. The real cause: no timing cache, and `builderOptimizationLevel` left at the maximum. Every CI build re-benchmarked every tactic for every layer from scratch. Introducing a persistent timing cache — keyed by TensorRT version and GPU, restored at the start of each CI run — cut rebuilds to about 6 minutes because unchanged layers reused their measured timings. They kept optimization level 5 only for the release build and dropped iteration builds to level 3. The lesson: the builder is a compiler, and like any compiler it has an incremental-build story; a timing cache is `ccache` for TensorRT.

### 4. The dynamic-shape throughput lie

A serving team reported their TensorRT engine was "20% slower than vLLM" on the same hardware and nearly switched stacks. Profiling told a different story. Their build set `optShapes` to batch 1 — the shape their latency benchmark used — while production ran a batch-32 in-flight batch. Every kernel in the engine had been autotuned for batch 1, so the batched decode ran on tactics chosen for a workload that did not exist in production. Rebuilding with `optShapes` at batch 32 — the real serving shape — flipped the comparison: the TensorRT engine became the faster of the two. The lesson: the optimization profile's `opt` shape is a performance contract; benchmark and build at the shape you actually serve.

### 5. The plugin version skew

A model relied on a custom TensorRT plugin — a hand-written CUDA kernel for an op the ONNX parser did not support — compiled against TensorRT 8.6. The team upgraded the serving container to TensorRT 9.x for an unrelated feature. The service segfaulted on the first inference. The wrong first hypothesis was a CUDA driver mismatch. The actual cause: the plugin was a shared library compiled against the 8.6 `IPluginV2` ABI, and the 9.x runtime expected a newer plugin interface; the ABI break manifested as a segfault deep in the plugin call. The fix was to recompile the plugin against the 9.x headers and add a CI check that plugin and TensorRT versions move together. The lesson: a plugin is native code linked against TensorRT's ABI — it is part of the version-pinning contract, not an independent artifact.

### 6. The KV cache OOM under load

A TensorRT-LLM deployment ran fine in staging and OOM-crashed in production within minutes of real traffic. Staging traffic was short prompts and short generations; production had long chat histories. The wrong hypothesis was that production simply needed a bigger GPU. The real cause: an older engine configuration using contiguous per-sequence KV allocation sized for `max_seq_len`. Under real concurrency the cache fragmented and each long sequence reserved a full max-length buffer regardless of actual length. Rebuilding with paged KV cache and `enable_block_reuse` turned the same GPU into one that comfortably served the production concurrency, because blocks were allocated to actual length and the long shared system prompt was deduplicated across requests. The lesson: KV memory pressure is an allocation-strategy problem before it is a hardware problem.

### 7. FP8 on the wrong GPU

A team built an FP8 TensorRT-LLM engine, benchmarked it on an H100, saw a 1.9× throughput win over FP16, and rolled it to a fleet that was a mix of H100 and A100 nodes. The A100 nodes served correctly but showed *no* speedup — in fact a slight regression. The wrong hypothesis was a misconfiguration on those nodes. The actual cause: FP8 tensor cores exist only on compute capability 8.9+; the A100 (8.0) has none, so FP8 matmuls fell back to a slow emulated path while still paying conversion overhead. The fix was to treat FP8 and FP16 engines as separate artifacts routed by GPU SKU, and to fail the build if an FP8 engine was scheduled onto pre-Hopper hardware. The lesson: a precision format is a hardware capability, not a universal speedup; FP8 is a Hopper-and-newer feature.

### 8. The fused layer that hid a NaN

A model started emitting NaNs after a precision change, and the team could not localize the failure. They added print statements around the suspect layer and found nothing — the intermediate tensor they wanted to inspect did not exist. The cause: the conv-bias-activation chain they suspected had been fused into a single CBR kernel, so the post-conv pre-activation tensor lived only in registers and was never written to memory the debugger could read. The fix was to rebuild with precision constraints that forced that layer chain unfused, expose the intermediate as a debug tensor with `--markDebug`, and confirm the conv output was overflowing FP16 range — a missing input normalization upstream. Once found, they restored fusion and fixed the normalization. The lesson: fusion is a compiler optimization that erases intermediates; debugging a fused engine sometimes means temporarily de-optimizing it.

### 9. The context that served one request at a time

A vision service running a TensorRT engine through a custom Python server hit a throughput ceiling far below what the GPU could do — GPU utilization sat around 30% under load that should have saturated it. The wrong hypothesis was that the engine needed a bigger batch size. The actual cause: the server created a single `IExecutionContext` at startup and guarded every `execute` call with a global lock, because someone had read that contexts are not thread-safe and concluded the safe thing was to serialize. Correct that contexts are not thread-safe — but the answer is one context *per worker thread*, each with its own CUDA stream, not one context behind a lock. Creating a pool of 4 contexts and 4 streams let the GPU scheduler overlap kernels from different requests and tripled throughput on the same hardware. The lesson: TensorRT's thread-safety rule tells you to *replicate* the context, not to *serialize* on it.

### 10. The dynamic batcher that never batched

A team deployed a detector on Triton, enabled dynamic batching, and saw no throughput improvement over batch-1 serving. They suspected the dynamic batcher was broken. It was not. Two settings defeated it. `max_queue_delay_microseconds` was left at 0, so Triton never waited to assemble a batch — every request went straight through alone. And the engine itself had been built with a static batch dimension of 1, so even when Triton *did* form a batch it could not be fed to an engine that only accepted batch 1. The fix was a pair: rebuild the engine with a dynamic batch dimension and an optimization profile whose `opt` shape was batch 16, and set `max_queue_delay_microseconds` to 3000. Throughput rose 5×. The lesson: dynamic batching is a contract between the server and the engine — both sides must agree that batches larger than 1 are legal and worth waiting for.

### 11. The build that was not reproducible

A team noticed their TensorRT engine's latency drifted between builds of the *same* model — sometimes 4.1 ms, sometimes 4.6 ms — with no code change. They suspected measurement noise. The real cause was the autotuner itself: kernel timing is a physical measurement, and on a build machine that was also running other GPU jobs, the tactic timings were contaminated. Sometimes the truly-fastest tactic lost a noisy race to a slightly-slower one, and the builder baked in the loser. The fix had two parts: run builds on a dedicated, otherwise-idle GPU so timings are clean, and commit a timing cache so that once a good set of tactics is found it is reused rather than re-raced. Latency stabilized at 4.1 ms. The lesson: the builder's autotuning is empirical, and empirical measurements need a controlled environment — a noisy build host produces a noisy engine.

### 12. The accuracy drop nobody could reproduce

A team's INT8 engine passed accuracy validation in CI but underperformed in production, and no one could reproduce the gap locally. The wrong hypothesis was a data pipeline difference between environments. The actual cause was the calibration cache. CI restored a `calib.cache` from a previous run to skip the slow calibration pass — good practice — but that cache had been generated from an *older* version of the calibration dataset, before the team added a new traffic segment. The engine's INT8 scales were therefore tuned for a distribution that no longer matched production. Deleting the stale cache, recalibrating against the current dataset, and keying the cache file by a hash of the calibration data fixed it. The lesson: a calibration cache is a function of the calibration data, and reusing it across data changes silently ships scales for the wrong distribution — version it like any other build input.

## When to reach for TensorRT / when not to

### Reach for TensorRT when

- You serve a **stable model on a fixed GPU SKU** at meaningful volume — the build cost amortizes over millions of inferences and the GPU-specific tuning pays off.
- You are **latency-bound on a CNN, vision transformer, or detector** — fusion and autotuning routinely deliver 2–5× over an eager framework runtime, and FP16/INT8 add more.
- You serve **LLMs in production** and need maximum tokens/s per GPU — TensorRT-LLM's in-flight batching, paged KV cache, and FP8 are at the frontier of inference throughput.
- You can afford a **proper build pipeline** — version-pinned, timing-cached, GPU-matched CI that produces immutable engine artifacts.
- You have an **accuracy validation harness** you can point at the built engine, so a precision change is a measured decision, not a hope.

### Skip TensorRT (or wait) when

- The model **changes constantly** — every change is a 10–40 minute rebuild, and a research or rapidly-iterating workload will spend more time building than serving.
- Your model is **full of unsupported or custom ops** — if most of the graph falls back to plugins or to the framework, you get the build cost without the fusion benefit.
- You serve on a **heterogeneous or unpredictable GPU fleet** — engine non-portability turns every SKU into a separate build-and-validate matrix; a portable runtime may be the better tradeoff.
- **Throughput is already adequate** — if you are well within latency budget and GPUs are not the cost driver, the engineering cost of a TensorRT pipeline is not repaid.
- You need **one artifact that runs everywhere** for distribution — that is what ONNX Runtime or a portable runtime is for; TensorRT is specialization, and specialization is the opposite of portability.

The through-line of this entire article is the compiler framing. TensorRT trades generality for speed: it gives up portability, build-time cheapness, and debuggability, and in exchange it hands you a program specialized so tightly to your GPU, your precision policy, and your shape regime that a general runtime cannot match it. Every failure mode in the case studies is the bill for that trade coming due in a place the team did not expect. Know that you are running a compiler, design the pipeline around a compiler, and TensorRT stops being a black box with a magic knob and becomes what it actually is — a predictable, powerful, and unforgiving optimization tool.

## Further reading

- [CUDA Graphs: eliminating launch overhead](/blog/machine-learning/deep-learning/cuda-graph) — the complementary technique TensorRT-LLM uses internally for decode replay.
- [Quantization at the edge: INT8, FP16, INT4 tradeoffs](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) — a deeper look at the numeric formats behind TensorRT's precision modes.
- [Optimizing LLM inference: a complete guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) — the broader serving picture that TensorRT-LLM slots into.
- [Choosing a GPU for LLM serving](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) — the cost and capacity side of the multi-GPU decisions in section 9.
- NVIDIA TensorRT Developer Guide and the TensorRT-LLM repository — the authoritative, version-specific reference for the APIs shown here.
