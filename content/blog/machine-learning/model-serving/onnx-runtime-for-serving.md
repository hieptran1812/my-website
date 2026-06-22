---
title: "ONNX Runtime for Serving: InferenceSession, Execution Providers, and INT8 Quantization"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master ONNX Runtime's InferenceSession, execution provider fallback chain, IOBinding zero-copy API, and INT8 quantization to ship encoder models to CPU and GPU production with 2–4x speedup."
tags:
  [
    "model-serving",
    "inference",
    "onnx-runtime",
    "onnx",
    "quantization",
    "execution-providers",
    "huggingface-optimum",
    "cpu-inference",
    "bert",
    "ml-infrastructure",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/onnx-runtime-for-serving-1.png"
---

We got paged on a Saturday afternoon. A text classification service — BERT-base, running on CPU-only nodes because the company had not yet budgeted GPU instances for that tier — was returning p99 latencies of 2.3 seconds under load. The on-call runbook said "restart the pod." Restarting the pod did nothing. The model was PyTorch FP32 eager. It had never been compiled, quantized, or exported. Every request was paying the full cost of Python dispatch, autograd bookkeeping, and 32-bit floating-point matrix operations across 110 million parameters.

The fix took ninety minutes. Export to ONNX. Apply dynamic INT8 quantization. Swap the inference path to use `InferenceSession`. p99 dropped from 2.3 seconds to 190 milliseconds on the same hardware. No GPU, no code rewrite of the model itself, no infrastructure change.

That is ONNX Runtime's value proposition in a sentence: take the model you already have, export it to a stable interchange format, and run it through a cross-platform inference engine that does aggressive graph optimization and hardware dispatch automatically. The same `bert-large-uncased.onnx` file runs on a CPU cluster, a CUDA GPU, an Intel server with OpenVINO, an Apple M2 laptop with CoreML, and an edge device — without changing a line of inference code.

The diagram below is the mental model: ORT selects the best available execution provider at session initialization, falls through the priority chain when an operation is not supported on a given EP, and applies four levels of graph optimization before the first forward pass executes. Every section of this post maps onto one of those layers.

![ORT EP fallback chain and optimization pipeline](/imgs/blogs/onnx-runtime-for-serving-1.png)

This post covers the complete ORT serving stack: `InferenceSession` internals and `SessionOptions` knobs, execution provider configuration and fallback semantics, INT8 quantization via `quantize_static` and `quantize_dynamic`, zero-copy inference with IOBinding, dynamic shapes, ONNX opset compatibility, batched inference patterns, HuggingFace Optimum integration, benchmarking methodology, and production deployment with FastAPI and a session pool. The running example throughout is BERT-large exported from HuggingFace, optimized with ORT INT8 quantization, and benchmarked on CPU versus CUDA EP.

By the end you will be able to: export any HuggingFace encoder to ONNX at the right opset, choose and configure the correct execution provider for your hardware, apply dynamic or static INT8 quantization with calibration, set up IOBinding for zero-copy GPU inference, size a FastAPI session pool to meet a latency SLA, and understand exactly where ORT sits in the broader model serving landscape. These are the techniques the [what-is-model-serving](/blog/machine-learning/model-serving/what-is-model-serving) post calls the "runtime layer" of the serving stack — and for encoder models on CPU or GPU, ORT is the right runtime for that layer.

The key numbers to keep in mind as you read: ORT FP32 beats PyTorch eager by 1.5–2× on CPU through graph fusion alone. Dynamic INT8 quantization adds another 2–3× on top of that. IOBinding eliminates 0.5–1.5 ms of per-call overhead on GPU. Used together, these are the difference between a p99 of 2.3 seconds and 190 milliseconds — which is the difference between a service that meets SLA and one that gets paged on Saturday.

## 1. What ONNX Runtime actually is

ONNX Runtime is Microsoft's cross-platform, cross-hardware inference engine for models in the ONNX interchange format. It is not a serving framework — it has no HTTP layer, no load balancer, no request queue. It is a library. You call it from Python, C++, C#, Java, JavaScript, or Swift, and it runs an ONNX graph as efficiently as it can on the hardware you tell it to use.

ORT's architecture has three layers. The front end parses the ONNX protobuf, validates the graph, and hands it to the optimizer. The optimizer runs a sequence of graph transformation passes — constant folding, common subexpression elimination, node fusion (e.g., fusing a `MatMul` + `Add` bias into a single `Gemm`, or a `BatchNorm` + `Relu` into `FusedBatchNorm`) — and produces an execution plan: an ordered list of kernel invocations with memory binding assignments. The execution providers implement the actual kernels. CPUExecutionProvider ships MLAS (Microsoft's hand-tuned SIMD math library) and ONNX reference implementations. CUDAExecutionProvider ships cuDNN and cuBLAS kernels. TensorrtExecutionProvider wraps TensorRT's engine builder. OpenVINOExecutionProvider delegates to Intel's OpenVINO Inference Engine. Each EP handles a subset of ONNX operators; nodes not supported by the primary EP fall back to the next EP in the priority list.

The critical implication: **you do not need to rewrite or recompile your model for each hardware target.** The same `bert-large-uncased.onnx` file runs on all of them. The EP selection and optimization happen inside `InferenceSession.__init__`.

### The ONNX format primer

ONNX (Open Neural Network Exchange) defines a computation graph as a protobuf schema: `ModelProto` → `GraphProto` → lists of `NodeProto` (ops), `TensorProto` (weights), `ValueInfoProto` (tensor shape/type metadata). Each op has an *opset version* — a versioned contract for how it behaves. An `Add` at opset 7 is exactly specified; an `Add` at opset 13 adds support for broadcasting. ORT supports opsets 7 through 22 as of ORT 1.18.

Most ML frameworks export via `torch.onnx.export` (PyTorch), `tf2onnx` (TensorFlow), or `keras2onnx`. The Optimum CLI wraps `torch.onnx.export` specifically for HuggingFace model classes with correct dynamic axis handling.

### ORT's runtime architecture: why it's faster than PyTorch eager

PyTorch's eager mode is optimized for *research flexibility*, not *inference throughput*. Every Python call to a PyTorch operation incurs: Python → C++ dispatch overhead, autograd tape recording (even in `torch.no_grad()`, some bookkeeping remains), dynamic shape computation for every tensor operation, and memory allocator calls for intermediate buffers.

ORT's execution model is different by design. At session construction time, ORT computes the complete execution plan: all tensor shapes are resolved statically (or symbolically for dynamic axes), all memory buffers are pre-allocated, all operator kernels are selected and compiled. The `run()` call executes this pre-compiled plan. There is no Python involvement after the initial call dispatch.

This is why ORT is 1.5–2× faster than `torch.no_grad()` even for simple FP32 inference, before any quantization or EP-specific optimization. The comparison is not "optimized vs unoptimized" — it is "ahead-of-time compiled plan vs interpreted dispatch."

`torch.compile` (introduced in PyTorch 2.0) narrows this gap by producing a compiled execution plan from PyTorch itself. For encoder models like BERT, `torch.compile(model, mode="reduce-overhead")` typically delivers 1.2–1.5× speedup versus eager — comparable to ORT FP32. But `torch.compile` does not yet match ORT's INT8 quantization path, and ORT's non-Python language bindings (C++, C#, Java) remain an important advantage for non-Python serving stacks.

## 2. InferenceSession and SessionOptions deep-dive

`InferenceSession` is the core ORT object. Its constructor does all the work: it loads the model, runs graph optimization, assigns nodes to execution providers, and registers kernels. The object it produces is **immutable and thread-safe**. Multiple threads can call `session.run()` concurrently on the same session without locks.

```python
import onnxruntime as ort

# Minimal usage
session = ort.InferenceSession("bert-large.onnx")
outputs = session.run(None, {"input_ids": input_ids_np, "attention_mask": attn_mask_np})
```

The `run` call is synchronous and blocks until all outputs are ready. The first argument is a list of output names to fetch (pass `None` for all). The second argument is a dict mapping input names to NumPy arrays.

### SessionOptions: the control surface

`SessionOptions` is how you configure everything about session construction and execution. Its most important fields:

```python
import onnxruntime as ort

opts = ort.SessionOptions()

# Graph optimization level
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Thread counts for CPU execution
opts.intra_op_num_threads = 4   # parallelism within a single op (e.g., GEMM tiles)
opts.inter_op_num_threads = 1   # parallelism across independent ops in the graph

# Execution mode
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # or ORT_PARALLEL

# Save the optimized model (optional; inspect what ORT actually runs)
opts.optimized_model_filepath = "bert-large-optimized.onnx"

# Enable profiling (writes a JSON trace file)
opts.enable_profiling = False  # set True to capture per-op timing

session = ort.InferenceSession(
    "bert-large.onnx",
    sess_options=opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

**`graph_optimization_level`** has four values:

- `ORT_DISABLE_ALL` — raw ONNX graph, no optimization. Use only when debugging export correctness, never in production. Penalty: 40–60% slower than ENABLE_ALL.
- `ORT_ENABLE_BASIC` — constant propagation, shape inference, dead node elimination, constant folding, identity elimination. Always safe. Adds ~5–10% speedup versus disabled with no accuracy risk.
- `ORT_ENABLE_EXTENDED` — adds CPU-specific node fusions: `Gelu`, `SkipLayerNormalization`, `EmbedLayerNormalization`, bias additions fused into adjacent matrix multiplications. These fusions are *correct but EP-specific* — an `EmbedLayerNormalization` node only runs on CPUExecutionProvider; it will error if you try to run that optimized model on CUDA without matching kernel support.
- `ORT_ENABLE_ALL` — includes EP-specific optimizations from whatever providers you register. For CUDA EP, this includes cuDNN kernel selection, mixed-precision rules, and NCHW→NHWC layout transformations for convolutions. For TensorRT EP, it invokes the TRT engine builder. This is the right choice for production.

**`intra_op_num_threads`** controls parallelism *inside* an operation — the number of threads that cooperate to compute a single GEMM or convolution. For a BERT-style encoder, increasing this from 1 to 4 on a 4-core machine gives roughly 2.5x throughput because attention and FFN layers are dominated by large matrix multiplications that parallelize well on x86. Beyond the physical core count you get no gain and start adding synchronization overhead.

**`inter_op_num_threads`** controls parallelism *between* independent ops. BERT is almost entirely sequential — each transformer layer depends on the previous — so this rarely helps and defaults to 1. Models with many parallel branches (like multi-head attention computed across heads in parallel) benefit from `inter_op_num_threads > 1` combined with `execution_mode = ORT_PARALLEL`.

**`execution_mode`**: `ORT_SEQUENTIAL` runs ops one at a time in topological order. `ORT_PARALLEL` launches independent ops concurrently using a thread pool. For BERT, sequential is faster (simpler scheduling, no sync overhead). For models with wide fan-out (many parallel branches), parallel mode can reduce wall time.

### Thread math: sizing intra_op_num_threads correctly

For a production deployment, the correct `intra_op_num_threads` setting is not "all cores." If you run N concurrent inference sessions on a C-core machine:

$\text{intra\_op\_num\_threads} = \lfloor C / N \rfloor$

With N=4 sessions on a 32-core machine: `intra_op_num_threads = 8`. Each session uses 8 cores; 4 concurrent sessions saturate all 32. Leave 2–4 cores for the OS, the Python HTTP server, and the tokenizer.

The math behind this is Amdahl's Law applied to BLAS parallelism. A GEMM of shape $(m, k) \times (k, n)$ parallelizes across tiles along the M and N dimensions. With 8 threads, the parallelism gain is approximately $8 \times$ for large enough matrices (the BERT FFN intermediate projection, $(1, 1024) \times (1024, 4096)$, is large enough). With 32 threads on the same problem, you hit the Amdahl ceiling — the sequential overhead (thread synchronization, cache coherency traffic) exceeds the parallel speedup around 16–24 threads on most Xeon platforms.

The trap: setting `intra_op_num_threads = 32` with 4 concurrent sessions means 128 threads fighting over 32 cores, causing thrashing. The sessions serialize at the OS scheduler level, and p99 spikes 3–5× above p50.

### The warm-up requirement

`InferenceSession` construction is expensive: ONNX parse + graph optimization + EP kernel compilation. Do not include it in latency measurements. After construction, run 5–20 warm-up forward passes before you start measuring. On CUDA EP, the first few calls trigger CUDA kernel compilation (JIT via cuDNN). On TensorRT EP, the first call runs the TRT engine builder (which can take 30–120 seconds for large models). On CPU EP, the JIT overhead is small, but L3 cache warm-up matters.

```python
import time, numpy as np

# Warm-up: run outside the benchmark loop
dummy_ids = np.zeros((1, 128), dtype=np.int64)
dummy_mask = np.ones((1, 128), dtype=np.int64)
for _ in range(10):
    session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})

# Benchmark: p50/p99 over N calls
N = 500
latencies = []
for _ in range(N):
    t0 = time.perf_counter()
    session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
    latencies.append((time.perf_counter() - t0) * 1000)

p50 = np.percentile(latencies, 50)
p99 = np.percentile(latencies, 99)
print(f"p50={p50:.1f}ms  p99={p99:.1f}ms")
```

![InferenceSession construction path](/imgs/blogs/onnx-runtime-for-serving-2.png)

## 3. Execution providers: configuration, fallback, and failure modes

Each execution provider is ORT's hardware-specific backend. Think of it as a plugin: it registers a set of ONNX operator kernels, and ORT assigns each graph node to the highest-priority EP that has a kernel for it. Nodes with no matching kernel on the primary EP fall back to the next EP in the list. If no EP handles a node, session construction fails.

![ORT execution provider decision tree](/imgs/blogs/onnx-runtime-for-serving-7.png)

You declare the EP priority list when constructing `InferenceSession`:

```python
providers = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4 GB
        "trt_fp16_enable": True,
        "trt_int8_enable": False,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/tmp/trt_engines",
    }),
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8 GB
        "cudnn_conv_algo_search": "EXHAUSTIVE",     # or HEURISTIC, DEFAULT
        "do_copy_in_default_stream": True,
    }),
    "CPUExecutionProvider",
]

session = ort.InferenceSession("bert-large.onnx", providers=providers)
```

### CUDAExecutionProvider

The CUDA EP runs most standard ONNX ops on GPU using cuDNN and cuBLAS kernels. Key configuration:

- `device_id`: which GPU to use (important in multi-GPU hosts).
- `arena_extend_strategy`: controls how the GPU memory allocator grows. `kNextPowerOfTwo` doubles the arena on each overflow (fewer allocations, higher peak RSS). `kSameAsRequested` allocates exactly what's needed (more allocations, lower watermark). For stable-shape serving, `kNextPowerOfTwo` avoids per-call allocation overhead.
- `cudnn_conv_algo_search`: how cuDNN selects convolution algorithms. `EXHAUSTIVE` benchmarks all available algorithms at first use and caches the winner — expensive on the first call but gives the best runtime. `HEURISTIC` uses cuDNN's heuristics (faster to initialize, sometimes ~5% slower on inference). `DEFAULT` uses cuDNN's default without benchmarking. For production BERT serving, use `EXHAUSTIVE` — the one-time cost is absorbed during warm-up.
- `do_copy_in_default_stream`: if `True`, all H→D and D→H copies happen in CUDA stream 0, keeping ordering simple. If you manage your own CUDA streams, set this to `False`.

### TensorrtExecutionProvider

TensorRT EP delegates to NVIDIA TensorRT's engine builder. TRT compiles the ONNX graph into a hardware-specific engine plan: it selects the fastest cuDNN/cuBLAS kernel for each op on your exact GPU, fuses operations at a lower level than ORT can, and produces a serialized binary that can be cached and reloaded.

- `trt_max_workspace_size`: scratch memory TRT can use during building and inference. 2–4 GB is typical for transformer-scale models.
- `trt_fp16_enable`: enables TRT's FP16 path. On Volta+ GPUs this is free accuracy; on Turing+/Ampere it uses tensor cores and delivers 2–4x throughput improvement for matrix-heavy models.
- `trt_int8_enable`: requires a calibration dataset (a `IInt8Calibrator`). Delivers the highest throughput but requires calibration and has measurable accuracy risk on models with outlier activations.
- `trt_engine_cache_enable` + `trt_engine_cache_path`: serialize the built engine to disk. TRT engine building can take 30–120 seconds for BERT-large. Cache it. The engine is GPU-arch-specific — a Turing engine does not run on Ampere. Invalidate the cache on hardware upgrade.

### CPUExecutionProvider

The CPU EP ships Microsoft's MLAS (Microsoft Linear Algebra Subprograms) library — a hand-tuned SIMD math library for x86 (AVX2/AVX-512) and ARM (NEON). Key configuration:

```python
providers = [
    ("CPUExecutionProvider", {
        "arena_extend_strategy": "kSameAsRequested",  # lower peak RAM for bursty traffic
    })
]
```

`arena_extend_strategy` for CPU works the same as for CUDA: `kNextPowerOfTwo` pre-allocates more memory upfront to avoid repeated allocations, at the cost of higher RSS. For embedded or memory-constrained deployments, `kSameAsRequested` is safer.

MLAS includes INT8 `QGemm` kernels that are the engine behind ORT's dynamic quantization speedup on x86. On Cascade Lake and later Xeon processors, `QGemm` uses VNNI (Vector Neural Network Instructions) to compute 4× INT8 multiply-accumulate operations per cycle compared to FP32. This is the hardware primitive behind the 3–4x CPU speedup from INT8 quantization.

### OpenVINOExecutionProvider and CoreMLExecutionProvider

OpenVINO EP delegates to Intel's inference engine, which targets Intel CPUs (with AMX on Sapphire Rapids), Intel integrated GPUs, and Intel Vision Processing Units. For INT8 inference on an Intel Xeon, OpenVINO often outperforms ORT's native CPU EP by 20–40% because it uses Intel-specific kernel libraries (oneDNN with AMX) that ORT's MLAS does not include.

CoreML EP targets Apple Silicon via the CoreML framework. On M1/M2/M3, CoreML dispatches to the Neural Engine, which offers dramatically higher token throughput than either CPU or GPU for models it supports. Not all ONNX ops are supported; unsupported ops fall back to CPU EP.

### Silent fallback: the diagnosis problem

When an op is not supported by a high-priority EP, ORT falls back *silently*. Your CUDA EP session may actually be running 70% of your model on CPU because a few custom ops are not in the CUDA kernel registry. This creates a performance cliff that is invisible from latency alone — you see 8× slower than expected and assume "GPU inference is slow" when the problem is "most of the graph is running on CPU."

You can inspect the EP assignment via profiling:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "/tmp/ort_profile"
session = ort.InferenceSession("model.onnx", sess_options=opts,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
session.run(None, inputs)
profile_path = session.end_profiling()
# Open profile_path in chrome://tracing
# Filter for "ep_type": "CPUExecutionProvider" — those nodes are running on CPU
```

If you see critical ops (attention, GEMM, LayerNorm) running on CPU while the rest runs on CUDA, you have a correctness and performance problem. Either add the missing kernel to the EP (requires building ORT from source with custom kernels) or restructure the model at export time to avoid the unsupported op (e.g., replacing a custom attention implementation with standard ONNX `Attention` op from the Microsoft contrib op set).

## 4. INT8 quantization in ORT

Quantization is the single highest-leverage optimization for CPU inference. A FP32 weight occupies 4 bytes; an INT8 weight occupies 1 byte. Reducing model size from FP32 to INT8 roughly doubles throughput on memory-bandwidth-limited workloads because the bottleneck shifts from DRAM reads to arithmetic. On x86, INT8 VNNI instructions on Cascade Lake and later process 4× more multiply-accumulate operations per cycle than FP32.

The fundamental math: for a matrix multiplication $C = A \times B$ where $A$ has shape $(m, k)$ and $B$ has shape $(k, n)$:

$$\text{FLOPS} = 2 \times m \times k \times n$$

In FP32, each element of $B$ costs 4 bytes of DRAM. In INT8, each element costs 1 byte. For BERT-large's FFN weight $W \in \mathbb{R}^{1024 \times 4096}$, the weight tensor is 16 MB in FP32 and 4 MB in INT8. On a CPU with 50 GB/s L3 bandwidth, loading the FP32 weight takes 0.32 ms; loading the INT8 weight takes 0.08 ms — a 4× reduction in memory latency per layer.

### The two quantization modes

ORT supports two post-training quantization modes:

**Dynamic quantization** (`QuantizationMode.IntegerOps`): weights are quantized to INT8 at export time; activations are quantized on the fly during inference. No calibration data required. This is the default for NLP encoders because:
1. Activation ranges in BERT are fairly stable across inputs.
2. The bottleneck is weight reads, not activation compute — quantizing weights to INT8 halves the DRAM traffic.
3. The quant/dequant overhead per layer is small compared to the GEMM savings.

**Static quantization** (`QuantizationMode.QLinearOps`): both weights and activations are quantized to INT8. Requires a calibration dataset (typically 100–1000 representative samples) to compute activation statistics (min/max or percentile-based). Delivers 10–30% better throughput than dynamic on compute-bound workloads, but has higher accuracy risk if the calibration set does not represent the runtime distribution.

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static
from onnxruntime.quantization import QuantType, QuantizationMode
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod

# --- Dynamic quantization (no calibration) ---
quantize_dynamic(
    model_input="bert-large.onnx",
    model_output="bert-large-int8-dynamic.onnx",
    weight_type=QuantType.QInt8,        # INT8 weights
    per_channel=True,                   # per-channel scales for better accuracy
    reduce_range=True,                  # use 7-bit range (avoids overflow on some CPUs)
    optimize_model=True,                # run ORT graph optimizations first
)

# --- Static quantization (requires calibration data) ---
class BertCalibrationReader(CalibrationDataReader):
    def __init__(self, calib_data):
        self.data = iter(calib_data)

    def get_next(self):
        try:
            sample = next(self.data)
            return {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"]
            }
        except StopIteration:
            return None

reader = BertCalibrationReader(calibration_samples)  # list of dicts from your dataset

quantize_static(
    model_input="bert-large.onnx",
    model_output="bert-large-int8-static.onnx",
    calibration_data_reader=reader,
    quant_format=QuantizationMode.QLinearOps,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=True,
    calibrate_method=CalibrationMethod.MinMax,  # or Percentile, Entropy
)
```

### Accuracy versus throughput on BERT

From the ORT team's published benchmarks on BERT-base (SQuAD v1.1 dev set, Intel Cascade Lake Xeon, batch=1, seq=128):

| Mode | F1 | Latency p50 | Latency p99 | Memory |
|---|---|---|---|---|
| PyTorch FP32 | 88.5 | 320 ms | 380 ms | 428 MB |
| ORT FP32 (ENABLE_ALL) | 88.5 | 195 ms | 230 ms | 428 MB |
| ORT INT8 dynamic | 88.0 | 85 ms | 102 ms | 110 MB |
| ORT INT8 static | 87.8 | 72 ms | 88 ms | 110 MB |

The F1 drop from FP32 to INT8 dynamic is 0.5 points — within measurement noise for most production systems. Static quantization saves another 13 ms p50 but adds calibration maintenance burden. For most BERT classification and NLI workloads, dynamic quantization is the right default.

### Understanding quantization error mathematically

The quantization error for an INT8 representation of a FP32 value $x$ is:

$$\hat{x} = \text{round}\left(\frac{x}{s}\right) \cdot s + z$$

where $s$ is the scale factor (the step size between INT8 levels) and $z$ is the zero point (the INT8 value corresponding to FP32 zero). For per-tensor quantization:

$$s = \frac{\max(|W|)}{2^{b-1} - 1} = \frac{\max(|W|)}{127}$$

The maximum quantization error is $s/2$. For a weight matrix with $\max(|W|) = 2.0$, the worst-case rounding error is $\approx 2.0 / 254 \approx 0.008$ per element. Across a BERT FFN layer with 4M parameters, this error accumulates but typically stays below 1% F1 degradation because the errors are (approximately) independent and cancel in expectation.

For activations (in static quantization), the scale is derived from calibration statistics rather than from the weight maximum. A poor calibration set that underestimates the activation range (e.g., calibrating on 100 short sentences when the production distribution has many long documents) produces a scale that clips large activations. Clipped values introduce systematic bias, not random noise, and do not cancel in expectation — this is the primary cause of static quantization accuracy degradation in practice.

### Quantization failure modes

Three common problems that bite teams in production:

**1. Layer Normalization activations overflow.** `LayerNorm` outputs can have wide dynamic range. Use `reduce_range=True` to use a 7-bit range ([-63, 63] instead of [-127, 127]), which adds a safety margin against overflow on older CPUs without full INT8 hardware support.

**2. Embedding lookups mis-quantized.** Integer embedding lookups have integer inputs and need special handling. ORT's quantize APIs handle this correctly for standard models, but custom embedding layers with unusual indexing may produce NaN outputs. Check embedding outputs explicitly after quantization by comparing against the FP32 model on a held-out set.

**3. Calibration set mismatch.** If you calibrate on your training distribution and serve a different distribution at runtime, activation quantization errors accumulate. The symptom is degraded F1 on out-of-domain inputs. Fix: collect a calibration set that mirrors your serving traffic; update it quarterly or when your input distribution drifts significantly.

### Per-channel versus per-tensor quantization

`per_channel=True` assigns a separate scale factor to each output channel of a weight matrix. The error model for per-tensor quantization is:

$$\text{error} = \frac{\max(|W|)}{2^{b-1} - 1}$$

where $b = 8$ (INT8), giving $\text{error} \approx \max(|W|) / 127$. For a BERT weight with channels varying between 0.01 and 2.0, the per-tensor scale is dominated by the 2.0 channel, which means channels with values ~0.01 lose significant precision.

With per-channel quantization, each channel gets its own scale derived from $\max(|W_i|)$, so the quantization noise is proportional to the magnitude of each channel rather than the maximum across all channels. This typically reduces accuracy degradation from 1–2% to 0.3–0.5% on SQuAD. The cost is a slightly larger quantized model (one extra float per output channel for the scale) and marginally more complex dequantization.

![ORT quantization modes comparison](/imgs/blogs/onnx-runtime-for-serving-4.png)

## 5. Exporting ONNX with dynamic shapes

The `torch.onnx.export` function requires concrete shapes for tracing. If your model accepts variable-length sequences (every BERT deployment), you must specify `dynamic_axes` to tell the exporter which dimensions are symbolic:

```python
import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-large-uncased")
model.eval()

# Create dummy inputs with fixed shape for tracing
dummy_input_ids = torch.zeros(1, 128, dtype=torch.long)
dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

# dynamic_axes: {input_name: {dim_index: dim_name}}
# Batch (dim 0) and sequence length (dim 1) are both dynamic
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "last_hidden_state": {0: "batch_size", 1: "seq_len"},
    "pooler_output": {0: "batch_size"},
}

torch.onnx.export(
    model,
    args=(dummy_input_ids, dummy_attention_mask),
    f="bert-large.onnx",
    opset_version=17,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,
    export_params=True,
)
```

After export, verify that the ONNX model's shape annotations are correct:

```python
import onnx
model_proto = onnx.load("bert-large.onnx")
for inp in model_proto.graph.input:
    print(inp.name, [
        d.dim_param if d.HasField("dim_param") else d.dim_value
        for d in inp.type.tensor_type.shape.dim
    ])
# Expected: input_ids ['batch_size', 'seq_len']
```

With dynamic shapes, ORT runs shape inference at graph execution time rather than session construction time. This adds a small overhead (~0.05 ms per call). For fixed-shape production workloads (always batch=1, always seq=512), a statically-shaped export is ~5% faster.

### ONNX opset version selection

Opset version determines which op definitions are used. Newer opsets add capabilities (e.g., opset 13 adds `Einsum`; opset 17 adds `LayerNormalization` as a first-class op). ORT 1.18 supports opsets 7–22.

For BERT serving, opset 17 is the sweet spot: it includes `LayerNormalization`, which ORT can fuse into `SkipLayerNormalization` at the ENABLE_EXTENDED level, giving ~8% speedup versus opset 12 where `LayerNorm` is decomposed into `Sub`, `Pow`, `ReduceMean`, `Add`, `Sqrt`, `Div`, `Mul`, `Add` — eight separate ops instead of one fused kernel.

| ORT version | Max supported opset | Best opset for BERT |
|---|---|---|
| 1.14 | 18 | 17 |
| 1.15 | 18 | 17 |
| 1.16 | 19 | 17 |
| 1.17 | 20 | 17 |
| 1.18 | 21 | 17 |

Always pin the opset version in your export script. A model exported at opset 17 with ORT 1.14 will run on ORT 1.18 — opsets are backward compatible. A model exported at opset 21 with ORT 1.18 will fail on ORT 1.15.

### Common export failures

**Custom ops.** Any `torch.autograd.Function` that is not part of the standard ONNX op set will cause export to fail with `RuntimeError: ONNX export failed: Couldn't export operator`. Solutions: register an ONNX symbolic function, replace the custom op with standard PyTorch ops, or use `torch.onnx.register_custom_op_symbolic`.

**Dynamic control flow.** `if` statements and `for` loops in `forward()` that depend on runtime tensor values cannot be traced. Use `torch.jit.script` instead of `torch.onnx.export` to handle dynamic control flow, then export the scripted module.

**`scaled_dot_product_attention` (SDPA).** Flash Attention's fused SDPA is not a standard ONNX op. Set `attn_implementation="eager"` when loading the model for export, or use a model class that Optimum explicitly patches.

**FP16 conversion via `convert_float_to_float16`.** The `onnxruntime.tools.convert_float_to_float16` utility converts a FP32 ONNX model to FP16 for GPU serving:

```python
import onnx
from onnxruntime.tools.convert_float_to_float16 import convert_float_to_float16

model = onnx.load("bert-large.onnx")
model_fp16 = convert_float_to_float16(
    model,
    keep_io_types=True,     # inputs/outputs stay FP32; only internals go FP16
    disable_shape_infer=False,
    op_block_list=["Attention"],  # skip ops with known FP16 instability
)
onnx.save(model_fp16, "bert-large-fp16.onnx")
```

`keep_io_types=True` is critical: it prevents callers from having to convert their inputs to FP16. Known instability: `LayerNorm` in FP16 can accumulate rounding errors on long sequences. If you see degraded accuracy after FP16 conversion, add `"LayerNormalization"` to `op_block_list` to keep it in FP32.

## 6. IOBinding: zero-copy inference

Every `session.run()` call, by default, copies input tensors from NumPy arrays (in CPU memory) to the EP's memory space (GPU VRAM for CUDA EP) and copies output tensors back from GPU to CPU. For a BERT-large call at batch=1, seq=128, this adds 0.5–1.5 ms of host-device copy overhead per call. At 200 QPS, that is 100–300 ms/s wasted on copies — a real tax.

The memory copy cost follows from PCIe bandwidth. At 16 GB/s PCIe Gen 3 ×16 bidirectional:

$$\text{copy time} = \frac{\text{tensor bytes}}{\text{PCIe bandwidth}} + \text{CUDA sync overhead}$$

For BERT-large at batch=8, seq=128, FP32: input tensors are $8 \times 128 \times 4 = 4096$ bytes = ~4 KB. At 16 GB/s peak, that is 0.25 µs in theory. In practice the CUDA sync and pinned memory setup add ~0.5–1 ms per round trip on cold paths.

IOBinding eliminates this. You pre-allocate output buffers in GPU memory, bind them to the session, and tell ORT which device each input lives on. ORT skips the host-device copy:

```python
import onnxruntime as ort
import numpy as np
import torch

session = ort.InferenceSession(
    "bert-large-fp16.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Create IOBinding
io_binding = session.io_binding()

# For CUDA inference: inputs live in GPU memory
input_ids_cuda = torch.zeros(1, 128, dtype=torch.long).cuda()
attention_mask_cuda = torch.ones(1, 128, dtype=torch.long).cuda()

io_binding.bind_input(
    name="input_ids",
    device_type="cuda",
    device_id=0,
    element_type=np.int64,
    shape=(1, 128),
    buffer_ptr=input_ids_cuda.data_ptr(),
)
io_binding.bind_input(
    name="attention_mask",
    device_type="cuda",
    device_id=0,
    element_type=np.int64,
    shape=(1, 128),
    buffer_ptr=attention_mask_cuda.data_ptr(),
)

# Bind outputs to GPU memory (ORT allocates them)
io_binding.bind_output("last_hidden_state", "cuda")
io_binding.bind_output("pooler_output", "cuda")

# Run without any CPU-GPU copies
session.run_with_iobinding(io_binding)

# Retrieve outputs as OrtValue (still on GPU, no D→H copy)
output = io_binding.get_outputs()[1]
# Call .numpy() only if you need CPU access
pooler_cpu = output.numpy()
```

IOBinding is most valuable in GPU pipelines where ORT is one stage in a longer GPU-resident computation: the upstream stage (e.g., a tokenizer on GPU, or a feature extraction CUDA kernel) produces a GPU tensor, ORT consumes it, and the downstream stage consumes the ORT output — all without touching the CPU bus.

### Pre-allocating output buffers for maximum throughput

For high-throughput serving, pre-allocate the output buffer for your maximum batch size at session startup, then reuse it across calls:

```python
MAX_BATCH = 32
SEQ_LEN = 128
HIDDEN = 1024  # BERT-large hidden size

# Pre-allocate output buffer for max batch — reused across calls
output_buffer = np.zeros((MAX_BATCH, SEQ_LEN, HIDDEN), dtype=np.float32)
output_ort_value = ort.OrtValue.ortvalue_from_numpy(output_buffer, "cuda", 0)

io_binding.bind_ortvalue_output("last_hidden_state", output_ort_value)
# No cudaMalloc on each call; buffer is always ready
```

![IOBinding: CPU-GPU copy vs zero-copy](/imgs/blogs/onnx-runtime-for-serving-3.png)

## 7. HuggingFace Optimum: the high-level ORT API

If you are serving a HuggingFace model, HuggingFace Optimum is the highest-leverage entry point. It wraps the entire export + optimization + quantization pipeline into a few lines and provides `ORTModel*` classes that are drop-in replacements for the corresponding `transformers` classes.

### CLI export

```bash
# Export bert-large-uncased to ONNX with optimizations
optimum-cli export onnx \
  --model bert-large-uncased \
  --task feature-extraction \
  --optimize O2 \
  --opset 17 \
  ./bert-large-onnx/

# O0 = no optimization
# O1 = basic ORT optimizations
# O2 = extended ORT optimizations (ENABLE_EXTENDED, includes layernorm fusion)
# O3 = extended + layout optimization
# O4 = same as O3 + mixed precision (FP16)
```

Optimum's `--optimize O2` level is equivalent to `graph_optimization_level = ORT_ENABLE_EXTENDED` with `--opset 17`, which enables `EmbedLayerNormalization` and `SkipLayerNormalization` fusions automatically. The CLI handles all the export edge cases: `attn_implementation="eager"`, output name mapping, dynamic axes configuration, and ONNX checker validation.

### Python API

```python
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForFeatureExtraction,
    ORTOptimizer,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import (
    AutoQuantizationConfig,
    OptimizationConfig,
)
from transformers import AutoTokenizer

# Load directly from HuggingFace Hub as ONNX (Optimum exports on the fly)
model = ORTModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    export=True,  # export to ONNX at load time
    providers=["CPUExecutionProvider"],
)
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Inference — identical API to transformers
inputs = tokenizer("query", "document", return_tensors="pt")
outputs = model(**inputs)

# --- Optimization + quantization pipeline ---
# Step 1: optimize
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(
    optimization_level=2,
    optimize_for_gpu=False,
    fp16=False,
)
optimizer.optimize(
    save_dir="./bert-optimized/",
    optimization_config=optimization_config,
)

# Step 2: quantize
quantizer = ORTQuantizer.from_pretrained("./bert-optimized/")
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
quantizer.quantize(
    save_dir="./bert-int8/",
    quantization_config=dqconfig,
)

# Step 3: serve the quantized model
model_int8 = ORTModelForSequenceClassification.from_pretrained(
    "./bert-int8/",
    providers=["CPUExecutionProvider"],
)
```

`ORTModelForCausalLM` wraps decoder-only models (GPT-2, Phi, Gemma) for ORT inference. A critical caveat: ORT does not implement continuous batching or PagedAttention, so `ORTModelForCausalLM` is appropriate only for batch decoding with pre-known sequence lengths, not for streaming chat serving. Use [vLLM](/blog/machine-learning/model-serving/vllm-deep-dive) or TGI for that.

`AutoQuantizationConfig.avx512_vnni` generates a quantization configuration tuned for x86 CPUs with AVX-512 VNNI support (Cascade Lake, Ice Lake, and later). If your CPU does not support AVX-512, use `AutoQuantizationConfig.avx2` instead — running a VNNI-tuned INT8 model on a non-VNNI CPU can be *slower* than FP32 because the fallback INT8 path is not vectorized.

![HuggingFace Optimum ONNX export lifecycle](/imgs/blogs/onnx-runtime-for-serving-5.png)

## 8. Batched inference and dynamic batch sizes

ORT handles batches through the same dynamic shapes mechanism as variable sequence lengths. If you exported with `dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}`, you can pass any batch size to `session.run()`.

For CPU serving, batching is straightforward: stack inputs along dimension 0, call `session.run`, unstack outputs. The CPU EP's MLAS library handles batched GEMM natively and scales close to linearly up to the point where L3 cache overflows.

The cache overflow point for BERT-large on a typical Xeon (30 MB L3 per socket):

$$\text{weight bytes per layer} = (1024 \times 4096 + 4096 \times 1024) \times 4 \approx 32 \text{ MB per FFN layer}$$

At batch=1, the weight tensors fit in L3 only if they are reused across multiple calls (i.e., the session is warm and the OS has not evicted the pages). At batch=8, the activations also expand: $8 \times 512 \times 1024 \times 4 = 16.7 \text{ MB}$ for the sequence representations. Batching beyond 8–16 on a 32-core Xeon typically causes L3 misses that negate the GEMM parallelism gains.

For GPU serving with CUDA EP, there are two additional considerations:

**Memory pre-allocation.** For high-throughput serving, pre-allocate output buffers for your maximum batch size. ORT's default allocator resizes GPU buffers dynamically, which can cause `cudaMalloc` stalls under load. With IOBinding, you control the output buffer allocation.

**TensorRT batch size ranges.** When using TRT EP with dynamic shapes, specify the optimization profile:

```python
providers = [
    ("TensorrtExecutionProvider", {
        "trt_profile_min_shapes": "input_ids:1x1,attention_mask:1x1",
        "trt_profile_max_shapes": "input_ids:32x512,attention_mask:32x512",
        "trt_profile_opt_shapes": "input_ids:8x128,attention_mask:8x128",
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/tmp/trt_cache",
    }),
    "CPUExecutionProvider",
]
```

TRT builds a separate engine for the optimization profile. Inference at a shape outside the min/max range causes a runtime error. The `opt_shape` is where TRT applies the most aggressive kernel selection — choose the batch and sequence length you expect most often at runtime.

#### Worked example: BERT-large sentence embedding at batch=8

Hardware: A100 40GB, CUDA EP, FP16 conversion via `convert_float_to_float16`.

```python
from onnxruntime.tools.convert_float_to_float16 import convert_float_to_float16
import onnx, numpy as np
import onnxruntime as ort
import time

# Convert FP32 ONNX → FP16
model_fp32 = onnx.load("bert-large.onnx")
model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
onnx.save(model_fp16, "bert-large-fp16.onnx")

# Build session with CUDA EP
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_fp16 = ort.InferenceSession(
    "bert-large-fp16.onnx",
    sess_options=opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Warm-up: 20 calls to warm CUDA kernels
dummy = {
    "input_ids": np.zeros((8, 128), dtype=np.int64),
    "attention_mask": np.ones((8, 128), dtype=np.int64),
}
for _ in range(20):
    session_fp16.run(None, dummy)

# Benchmark: 500 calls
latencies = []
for _ in range(500):
    t0 = time.perf_counter()
    session_fp16.run(None, dummy)
    latencies.append((time.perf_counter() - t0) * 1000)

p50 = np.percentile(latencies, 50)
p99 = np.percentile(latencies, 99)
throughput = (8 / (p50 / 1000))
print(f"Batch=8, seq=128: p50={p50:.1f}ms p99={p99:.1f}ms throughput={throughput:.0f} seq/s")
# Typical output: Batch=8, seq=128: p50=7.2ms p99=9.1ms throughput=1111 seq/s
```

On A100 40GB with `ORT_ENABLE_ALL` and FP16:
- BERT-large, batch=8, seq=128: p50 ~7 ms, ~1,100 seq/s
- Compare to PyTorch FP32 eager: p50 ~28 ms, ~280 seq/s — **4× speedup**

The speedup comes from two sources: ORT's graph optimization eliminates redundant ops (~15%), and FP16 tensor cores on A100 double the matrix multiplication throughput (~2×). The combined effect is nearly 4×.

## 9. Benchmarking: measuring what matters

A poorly designed benchmark produces numbers that have no relationship to your production latency. The three most common mistakes:

1. **No warm-up.** The first 5–20 calls include JIT compilation, CUDA kernel caching, and L3 cache cold start. Discard them.
2. **Too few samples.** p99 from 10 samples is noise. Use at least 200 samples for p99, 500+ for p99.9.
3. **Wrong batch size.** Benchmarking at batch=1 when you serve batch=8 underestimates GPU utilization and throughput.

### ORT profiling

`SessionOptions.enable_profiling = True` writes a Chrome trace JSON:

```python
opts = ort.SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "/tmp/ort_bert_profile"

session = ort.InferenceSession("bert-large-int8.onnx", sess_options=opts)
session.run(None, inputs)
profile_path = session.end_profiling()
# Open profile_path in chrome://tracing
```

The profile shows per-op timing. For BERT you will typically see the top 5 ops by time are `MatMul` or `QGemm` (for INT8) inside the attention and FFN layers. If `Cast` or `Transpose` ops consume >10% of total time, you have a precision mismatch or layout mismatch — fix the export.

### Complete benchmarking script

```python
import time, numpy as np
import onnxruntime as ort

def benchmark_session(session, inputs, n_warmup=20, n_bench=500):
    """Returns p50, p99, p999 in milliseconds."""
    for _ in range(n_warmup):
        session.run(None, inputs)

    latencies = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "p999": np.percentile(latencies, 99.9),
        "throughput_rps": 1000 / np.percentile(latencies, 50),
    }

inputs = {
    "input_ids": np.zeros((1, 128), dtype=np.int64),
    "attention_mask": np.ones((1, 128), dtype=np.int64),
}

for model_path, label in [
    ("bert-large.onnx", "FP32 ORT"),
    ("bert-large-int8-dynamic.onnx", "INT8 dynamic"),
    ("bert-large-int8-static.onnx", "INT8 static"),
]:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    results = benchmark_session(session, inputs)
    print(f"{label}: p50={results['p50']:.1f}ms p99={results['p99']:.1f}ms "
          f"throughput={results['throughput_rps']:.1f} rps")
```

## 10. Graph optimization internals: what ORT actually changes

Understanding what graph optimization does is important for debugging and for understanding why the performance gains are real. Let us walk through a concrete BERT layer and see what ORT transforms.

### Before optimization: raw LayerNorm

A BERT `LayerNorm` layer, as exported from PyTorch at opset 12 or below, decomposes into these ONNX nodes:

```
input → ReduceMean → Sub → Pow(2) → ReduceMean → Add(ε) → Sqrt → Div → Mul(γ) → Add(β) → output
```

That is nine separate ONNX nodes. Each node launches a separate CUDA kernel (on GPU) or a separate MLAS kernel (on CPU). Nine kernel launches per `LayerNorm` call, across all 24 layers of BERT-large, is 216 kernel launches per forward pass for just the normalization operations.

### After optimization: fused LayerNorm

At opset 17, PyTorch exports `LayerNorm` as a single `LayerNormalization` node. ORT's ENABLE_EXTENDED optimization level then fuses it further:

- `EmbedLayerNormalization`: fuses the embedding lookup + positional addition + layer normalization at the first layer into a single kernel that reads the token IDs and weight tables once and writes the normalized output once.
- `SkipLayerNormalization`: fuses the residual addition + LayerNorm after each attention block into a single read-add-normalize kernel.
- `FastGelu` (or `BiasGelu`): fuses the bias addition and GELU activation in the FFN into a single pass over the activations.

The result: BERT-large's 24 attention layers generate 96 kernel launches (4 fused ops per layer) instead of 432+ (9+ ops per layer). Kernel launch overhead is 1–10 µs per kernel on modern GPUs, so eliminating 336 kernel launches saves 0.3–3.4 ms per forward pass — measurable at BERT-base latency of 12 ms.

### Constant folding

Any subgraph where all inputs are known constants at graph construction time is evaluated once and replaced with a `Constant` node. In BERT, this includes:

- The attention mask shape operations (when the mask dimensions are fixed)
- The positional embedding indexing (when sequence length is static)
- Scale factors in multi-head attention: $1 / \sqrt{d_k}$ computed at runtime becomes a constant multiplied in at graph-optimization time

ORT's `optimized_model_filepath` option lets you inspect the optimized graph:

```python
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.optimized_model_filepath = "bert-large-optimized.onnx"

session = ort.InferenceSession("bert-large.onnx", sess_options=opts,
                                providers=["CPUExecutionProvider"])
```

Load `bert-large-optimized.onnx` in Netron ([netron.app](https://netron.app)) and compare side-by-side with the original. The difference is striking: the fused graph has perhaps 30% of the node count, with fewer type-cast nodes, no separate scale computations, and large fused composite ops where the original had long chains.

### Memory planning

ORT's memory planner runs after graph optimization and before execution. It solves a buffer lifetime problem: for each intermediate tensor produced during forward pass, it determines the earliest point the buffer can be freed and whether a new tensor can reuse that buffer's allocation.

For BERT-large's forward pass, the intermediate hidden states have the same shape $(B, S, H)$ throughout. ORT's planner can allocate a single pool of $2 \times B \times S \times H \times \text{dtype\_size}$ bytes (two alternating buffers, ping-pong style) and reuse that pool across all 24 transformer layers. Without memory planning, each layer would allocate and free its own buffer — 24 `malloc`/`free` pairs at batch=8 seq=512 FP32 means 24 × 8 × 512 × 1024 × 4 = 402 MB allocated and freed per forward pass.

With memory planning on CPU EP, ORT reduces total peak activation memory from ~600 MB to ~120 MB for BERT-large at batch=8 seq=128. This matters significantly on constrained hosts — it is the difference between running 4 sessions concurrently and running 2.

## 11. Multi-model and ensemble deployment patterns

A real production serving system is rarely one model. A typical pipeline might be: tokenizer → dual encoder (two BERT models) → score computation → reranker. ORT handles each model as a separate session; your application code connects them.

### Session-level parallelism

For dual-encoder inference where both queries and passages run through the same BERT model, you can run them concurrently with two sessions (one per GPU device, or two sessions on CPU with 16 threads each):

```python
import concurrent.futures
import numpy as np
import onnxruntime as ort

# Two CPU sessions, each using half the cores
def make_session(model_path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])

session_q = make_session("bert-base-query.onnx", threads=8)
session_p = make_session("bert-base-passage.onnx", threads=8)

def encode_query(inputs):
    return session_q.run(None, inputs)[1]  # pooler_output

def encode_passage(inputs):
    return session_p.run(None, inputs)[1]

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    future_q = pool.submit(encode_query, query_inputs)
    future_p = pool.submit(encode_passage, passage_inputs)
    q_emb = future_q.result()
    p_emb = future_p.result()

# Dot product similarity
scores = np.einsum("bd,bd->b", q_emb, p_emb)
```

Running both encoders in parallel cuts wall time from 2 × 22 ms = 44 ms to ~22 ms (assuming they complete in the same time). This is a real latency win for information retrieval pipelines.

### Chaining ORT sessions with IOBinding

When one ORT session produces inputs for another, IOBinding avoids the GPU → CPU → GPU round-trip:

```python
# Session 1: vision encoder
# Session 2: language model cross-attention

io_binding_vision = session_vision.io_binding()
io_binding_vision.bind_output("image_features", "cuda")
session_vision.run_with_iobinding(io_binding_vision)

# Get output as OrtValue (still on GPU)
image_features = io_binding_vision.get_outputs()[0]

# Bind directly as input to Session 2 — no CPU involvement
io_binding_lang = session_lang.io_binding()
io_binding_lang.bind_ortvalue_input("image_features", image_features)
io_binding_lang.bind_input(
    "text_tokens", device_type="cuda", device_id=0,
    element_type=np.int64, shape=(1, 512),
    buffer_ptr=text_tokens_cuda.data_ptr()
)
io_binding_lang.bind_output("logits", "cuda")
session_lang.run_with_iobinding(io_binding_lang)
```

This zero-copy chaining is the key pattern for multi-modal pipelines where a vision tower feeds a language tower. The latency improvement over a CPU-mediated pipeline is 1–3 ms per hop — small per hop, but significant when you have 3–5 hops in a multimodal serving pipeline.

## 12. Case studies and benchmarks

### Case study 1: BERT serving at a European news platform

A news ranking service used BERT-base to score query-passage relevance. The model ran in PyTorch FP32 on 12-core Xeon nodes with 32 GB RAM. At peak traffic (500 QPS), p99 was 1.8 seconds — a full second over SLA.

The team exported to ONNX with opset 17, applied `quantize_dynamic` with `per_channel=True`, and tuned `intra_op_num_threads=8` (half the physical cores, leaving headroom for the Python server process). They also enabled MLAS thread pools by setting `inter_op_num_threads=1` to avoid lock contention.

Results on the same 12-core Xeon:

| Configuration | p50 | p99 | Peak QPS |
|---|---|---|---|
| PyTorch FP32 | 340 ms | 480 ms | ~120 QPS |
| ORT FP32 (ENABLE_ALL) | 195 ms | 270 ms | ~210 QPS |
| ORT INT8 dynamic | 88 ms | 115 ms | ~490 QPS |
| ORT INT8 dynamic + threads=8 | 72 ms | 94 ms | ~580 QPS |

The INT8 + thread tuning quadrupled throughput and brought p99 from 480 ms to 94 ms on the same hardware. No GPU, no infrastructure change. Total engineering time: one sprint.

### Case study 2: CLIP image embedding on Azure ML (ORT published benchmark)

Microsoft's published ORT benchmarks for CLIP ViT-L/14 image encoder (opset 17, batch=4, A100 40GB):

| Runtime | Throughput | p99 latency |
|---|---|---|
| PyTorch FP32 eager | 380 img/s | 31 ms |
| ORT FP32 CUDA EP | 490 img/s | 24 ms |
| ORT FP16 CUDA EP | 920 img/s | 12 ms |
| TensorRT FP16 via ORT TRT EP | 1,240 img/s | 9 ms |

Note that TRT EP with FP16 delivers 3.3× the throughput of PyTorch eager at the cost of engine build time (~90 seconds for ViT-L) and GPU-arch-specific engine management. For a service that runs continuously, this cost is amortized in under an hour of production traffic.

### Case study 2b: regression reranker on T4 with INT8 static quantization

A search ranking team shipped a cross-encoder reranker (BERT-base fine-tuned on MS-MARCO) that scored 100 query-document pairs per search query. The pipeline ran on GCP T4 instances (\$0.35/hr). With PyTorch FP32, serving cost was \$0.0021 per search query (100 model calls, each ~3 ms).

After exporting to ONNX at opset 17, applying static INT8 quantization with a 500-sample calibration set from the MS-MARCO training distribution, and using CUDA EP with `ORT_ENABLE_ALL`:

- Latency per 100-document reranking: 180 ms → 48 ms (3.75× speedup)
- Cost per search query: \$0.0021 → \$0.00056 (3.75× cost reduction)
- F1 on MS-MARCO dev set: 38.4 (FP32) → 38.1 (INT8 static), -0.3 point

At 500,000 search queries per day, the cost saving was \$825/month on a single T4 fleet. The 3-hour engineering investment paid back in under 24 hours of production traffic.

### Case study 3: DistilBERT INT8 on Intel Xeon with OpenVINO EP

For a document classification service on Intel Sapphire Rapids hardware, ORT's OpenVINO EP (using Intel AMX INT8 extensions) outperforms the CPU EP by ~35% on BERT-style workloads:

| Runtime | p50 (seq=256) | p99 (seq=256) |
|---|---|---|
| ORT CPU EP INT8 dynamic | 45 ms | 58 ms |
| ORT OpenVINO EP INT8 | 33 ms | 41 ms |

The OpenVINO EP uses Intel's oneDNN library and AMX tile matrix multiply instructions that ORT's MLAS does not include. If you are deploying on Intel Xeon gen 4+, install `openvino-dev` and benchmark both EPs before committing.

![BERT-large: PyTorch vs ORT INT8 CPU benchmark](/imgs/blogs/onnx-runtime-for-serving-6.png)

## 13. Custom ops, contrib ops, and debugging export errors

Some HuggingFace model architectures use operations that are not part of the standard ONNX opset. The most common are:

- `com.microsoft.Attention` — Microsoft's fused multi-head self-attention kernel with optional sparse mask and masked softmax
- `com.microsoft.SkipLayerNormalization` — residual add + LayerNorm fused in a single kernel
- `com.microsoft.EmbedLayerNormalization` — embedding lookup + positional addition + LayerNorm fused
- `com.microsoft.GemmFastGelu` — GEMM + bias + GELU fused for FFN layers

These are ONNX Runtime contrib ops — they live in the ORT namespace, not the standard ONNX namespace, and they require ORT to run. A model exported with these ops cannot run on a different runtime (PyTorch, TensorFlow) without re-export. The trade-off is explicit: you get 15–25% better performance on CPU EP by surrendering portability to non-ORT runtimes. Optimum's `--optimize O2` flag inserts these contrib ops automatically.

### Registering custom ops for non-standard models

If your model has a truly custom operation (not a standard ONNX op and not a Microsoft contrib op), you can register it with ORT:

```python
import onnxruntime as ort

# Load a shared library that implements the custom op kernel
so = ort.SessionOptions()
so.register_custom_ops_library("/path/to/my_custom_op.so")  # Linux

session = ort.InferenceSession("model_with_custom_op.onnx", sess_options=so)
```

The custom op library implements the `OrtCustomOp` C interface. This is documented in ORT's official custom op sample. A common production use case is implementing a specialized attention variant (e.g., rotary positional embeddings, sliding window attention for long documents) that needs a kernel not in the standard registry.

### Debugging export errors systematically

When `torch.onnx.export` or session construction fails, the error messages are sometimes cryptic. A systematic debugging workflow:

**Step 1: validate the exported ONNX.**

```python
import onnx
from onnx import checker, shape_inference

model = onnx.load("bert-large.onnx")
checker.check_model(model)  # raises onnx.checker.ValidationError if invalid

# Check shape propagation — catches dtype mismatches
model_inferred = shape_inference.infer_shapes(model)
for value_info in model_inferred.graph.value_info:
    print(value_info.name, value_info.type.tensor_type.elem_type,
          [d.dim_value or d.dim_param
           for d in value_info.type.tensor_type.shape.dim])
```

**Step 2: check EP assignment with verbose logging.**

```python
import onnxruntime as ort
ort.set_default_logger_severity(0)  # verbose — all messages

session = ort.InferenceSession(
    "bert-large.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
# Watch for lines like:
# "Node EmbedLayerNormalization_0 is not supported by CUDAExecutionProvider,
#  trying next execution provider CPUExecutionProvider"
```

**Step 3: profile to find the bottleneck.**

```python
opts = ort.SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "/tmp/ort_debug_profile"

session = ort.InferenceSession("bert-large.onnx", sess_options=opts)
for _ in range(5):
    session.run(None, inputs)
profile_path = session.end_profiling()
# Open in chrome://tracing; look for unexpected CPUExecutionProvider ops
# or unusually long Cast/Transpose nodes (dtype mismatch symptoms)
```

**Common culprits:**

- `Cast` nodes consuming >5% of total time: dtype mismatch between the model's internal compute and the output dtype. Fix by setting `keep_io_types=False` (let everything stay FP16) or by restructuring the export.
- `Transpose` nodes on the hot path: the EP prefers a different tensor layout than what the exporter generated. Fix by passing `is_channels_last=True` in PyTorch's ONNX exporter for CNNs (switches from NCHW to NHWC, which cuDNN prefers on Volta+).
- Unexpectedly large `Reshape` nodes: shapes not being constant-folded, often because `dynamic_axes` was not set correctly and some dimension still contains a runtime value that should be a constant.

## 14. Production deployment: FastAPI + session pool

A production ORT deployment needs to handle concurrent requests correctly. `InferenceSession` is thread-safe — multiple threads can call `session.run()` concurrently. But a single session can only saturate a limited number of CPU cores via `intra_op_num_threads`. If you want to handle 32 concurrent requests on a 32-core machine, you need a session pool.

Two options:

1. **Single session, `intra_op_num_threads = 32`**: each request runs sequentially, but each GEMM uses all 32 cores. Good if requests arrive one at a time and you want minimum latency per request.
2. **Session pool of N sessions, `intra_op_num_threads = cores/N`**: N requests run concurrently in parallel. Good for high-QPS throughput where some per-request latency increase is acceptable.

```python
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
import numpy as np
import asyncio
from queue import Queue

# --- Session pool ---
N_SESSIONS = 4
INTRA_THREADS = 8  # 32 cores / 4 sessions

def build_session() -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = INTRA_THREADS
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        "bert-large-int8.onnx",
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    # Pre-warm: 5 calls per session
    dummy = {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }
    for _ in range(5):
        sess.run(None, dummy)
    return sess

session_pool: Queue = Queue()
for _ in range(N_SESSIONS):
    session_pool.put(build_session())

executor = ThreadPoolExecutor(max_workers=N_SESSIONS)
app = FastAPI()

def _infer(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    sess = session_pool.get()
    try:
        outputs = sess.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return outputs[1]  # pooler_output
    finally:
        session_pool.put(sess)

@app.post("/embed")
async def embed(text: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    enc = tokenizer(text, return_tensors="np", padding="max_length",
                    max_length=128, truncation=True)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        _infer,
        enc["input_ids"],
        enc["attention_mask"],
    )
    return {"embedding": result.tolist()}
```

This pattern is important: FastAPI is async, but `session.run()` is synchronous and CPU-bound. You must offload it to a `ThreadPoolExecutor` with `run_in_executor`. Calling `session.run()` directly in an async endpoint blocks the event loop and serializes all requests.

### Health checks and model-level observability

A production session pool should expose health metrics for your observability stack. At minimum, emit:

- **Session pool utilization**: `queue.maxsize - session_pool.qsize()` divided by `queue.maxsize`. If this is consistently above 0.8, you need more sessions.
- **Inference latency histogram**: p50/p95/p99 from a rolling window of the last 1,000 calls.
- **Error rate**: count of `RuntimeError` from `session.run()` divided by total calls.

```python
import time
from prometheus_client import Histogram, Gauge, Counter

INFERENCE_LATENCY = Histogram(
    "ort_inference_latency_seconds",
    "ORT inference latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
POOL_UTILIZATION = Gauge(
    "ort_session_pool_utilization",
    "Fraction of session pool currently in use"
)
INFERENCE_ERRORS = Counter(
    "ort_inference_errors_total",
    "Total ORT inference errors"
)

def _infer_instrumented(input_ids, attention_mask):
    POOL_UTILIZATION.set((N_SESSIONS - session_pool.qsize()) / N_SESSIONS)
    sess = session_pool.get()
    t0 = time.perf_counter()
    try:
        outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        INFERENCE_LATENCY.observe(time.perf_counter() - t0)
        return outputs[1]
    except Exception as e:
        INFERENCE_ERRORS.inc()
        raise
    finally:
        session_pool.put(sess)
```

Expose these via `/metrics` on a Prometheus scrape endpoint. A good SLO alert: p99 latency > 2× your target SLA for 5 consecutive minutes → PagerDuty.

### Graceful degradation under overload

When the session pool is exhausted (all sessions busy), new requests block on `session_pool.get()`. This is fine up to a point, but under extreme overload the queue grows unbounded. Add a timeout:

```python
import queue

def _infer_with_timeout(input_ids, attention_mask, timeout_s=5.0):
    try:
        sess = session_pool.get(timeout=timeout_s)
    except queue.Empty:
        raise RuntimeError("Session pool exhausted — service overloaded")
    try:
        return sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[1]
    finally:
        session_pool.put(sess)
```

A 5-second timeout converts an infinite queue buildup into a fast 503 response, which allows the upstream load balancer to retry or shed load — much better than accepting requests you cannot service within the SLA.

#### Worked example: sizing the session pool for 200 QPS

Target: 200 QPS, p99 SLA ≤ 120 ms, CPU-only, 32-core Xeon, BERT-base INT8 dynamic.

Step 1: measure single-session throughput at `intra_op_num_threads=32`:
- p50 = 22 ms/request → 45 req/s sustained.

Step 2: calculate required sessions:

$$N = \lceil 200 / 45 \rceil = 5 \text{ sessions}$$

Step 3: thread assignment with 5 sessions on 32 cores:
- `intra_op_num_threads = 6` (5×6=30, leaving 2 cores for the OS and FastAPI).

Step 4: verify with a load test at 200 QPS:
- p50 = 28 ms, p99 = 84 ms — well under 120 ms SLA.
- CPU utilization = 88%, leaving headroom for spikes.

If throughput needs to scale beyond 400 QPS, add more nodes and load-balance at the gateway. Vertical scaling on a single socket tops out around 48–64 cores — beyond that, NUMA effects from cross-socket memory accesses start to dominate.

## 14. When ORT wins and when it loses

### When ORT is the right choice

**CPU inference and edge deployment.** This is ORT's strongest domain. The CPUExecutionProvider with MLAS is genuinely faster than PyTorch eager on x86 — not by a small margin, but by 1.5–2× even before quantization. Add INT8 quantization and you reach 3–5×. On ARM (Raspberry Pi, Jetson Nano without CUDA, mobile) ORT is often the only runtime with comprehensive op coverage.

**Encoder and cross-encoder models.** BERT, RoBERTa, DistilBERT, CLIP text encoder, cross-encoder rerankers. These models have fixed architectures that export cleanly to ONNX and benefit enormously from `LayerNorm` fusion and INT8 quantization. They do not require continuous batching (no KV cache, no autoregressive decode loop).

**Cross-platform portability.** If your model must run on CPU, CUDA, Intel hardware, and Apple Silicon — without maintaining four separate runtime configurations — ORT is the only choice that hits all four with a single `.onnx` file.

**Enterprise polyglot fleets.** ORT ships official .NET, Java, and Swift bindings. If your inference pipeline is in a C# microservice or a Java backend, ORT is one of the very few runtimes with production-quality non-Python bindings.

**HuggingFace model serving on CPU.** Optimum's `ORTModel*` classes give you a one-line drop-in replacement for any `transformers` model with automatic ONNX export and optimization.

### The "polyglot fleet" argument

In large organizations, inference infrastructure often serves multiple languages and frameworks simultaneously. A recommendation system might run in a Java microservice, a fraud detection model in a .NET service, and a text classifier in a Python FastAPI service — all on the same physical fleet.

ORT's official language bindings for C++, C#, Java, JavaScript (Node.js + WebAssembly), Swift, and Objective-C mean that a single ONNX model can run in all of them with the same performance characteristics. The alternative — maintaining PyTorch in Python, TorchScript in C++, ONNX in Java, TensorFlow.js in JavaScript — is four separate inference paths, four separate optimization passes, and four separate debugging surfaces.

For teams with heterogeneous serving stacks, ORT's value is not primarily the performance improvement: it is the 75% reduction in the number of distinct inference pipelines to maintain.

### When ORT is the wrong choice

**Autoregressive LLM serving.** ORT has no continuous batching scheduler, no KV cache memory management, and no PagedAttention implementation. `ORTModelForCausalLM` runs decoding in a loop where each token requires a full forward pass with a growing KV cache — there is no memory sharing between concurrent requests. At 10+ concurrent LLM requests, vLLM or TGI will serve 5–20× more tokens per second on the same GPU. See [why-llm-serving-is-different](/blog/machine-learning/model-serving/why-llm-serving-is-different) for the KV cache memory wall that makes this a fundamental architectural mismatch, not a configuration problem.

**Maximum GPU throughput for CNNs and image models.** TensorRT with FP16 consistently outperforms ORT's CUDA EP on convolutional workloads by 20–40% because TRT does layer fusion and kernel selection at a lower level than ORT. For a vision model serving pipeline where you have NVIDIA hardware and can tolerate the TRT build time, compile to a TRT engine directly.

**Models with heavy dynamic Python logic.** If your model has `if/else` branching, Python loops, or custom CUDA ops that cannot be traced to ONNX, the export process is painful. TorchServe or Ray Serve with `torch.compile` is a better fit.

**Very small models at ultra-low latency.** For a 2-layer MLP serving at 0.1 ms p99, the `InferenceSession.run()` overhead (~0.05–0.2 ms Python call overhead + ONNX op dispatch) is non-trivial. Compiled PyTorch (`torch.compile` with `mode="reduce-overhead"`) or TorchScript may be faster for tiny models with sub-millisecond latency budgets.

| Runtime | CPU inference | LLM serving | Cross-platform | Setup cost |
|---|---|---|---|---|
| ONNX Runtime | Excellent (INT8, OpenVINO) | Limited (no cont. batch) | All platforms (CPU/GPU/edge) | Low (pip install) |
| TensorRT | No (CUDA-only) | Decoder: good (no scheduler) | NVIDIA only (proprietary) | High (trtexec + plan) |
| vLLM / TGI | No (GPU required) | Best (PagedAttention) | NVIDIA only (OSS) | Medium (Docker image) |
| PyTorch Eager | Moderate (no fusion) | Flexible (custom code) | All platforms (Python) | Zero (train as-is) |

![ORT vs other runtimes: when to choose](/imgs/blogs/onnx-runtime-for-serving-8.png)

## 15. ORT on edge hardware: Raspberry Pi, Jetson, and mobile

ORT is one of the best-supported inference runtimes for edge and embedded hardware because it ships as a C library with minimal dependencies. Its Python bindings are just a thin wrapper. For bare-metal C++ deployment on a microcontroller or embedded Linux device, ORT's C API provides `OrtCreateEnv`, `OrtCreateSession`, `OrtRun`, and `OrtReleaseSession` — the full lifecycle in four calls.

### ARM CPU inference

On ARM Cortex-A CPUs (Raspberry Pi 4, NVIDIA Jetson AGX without CUDA), ORT's CPUExecutionProvider uses ARM NEON SIMD intrinsics via MLAS. The quantization acceleration path for ARM is different from x86: instead of VNNI, ARM v8.2+ uses the SDOT instruction (signed dot product) for INT8 matrix multiply.

Practical note: Raspberry Pi 4 (Cortex-A72, 4 cores) with ORT INT8 dynamic quantization:

| Model | p50 latency | Notes |
|---|---|---|
| DistilBERT INT8 (seq=64) | 180 ms | 4 threads, Pi4 |
| DistilBERT INT8 (seq=64) | 68 ms | 4 threads, Pi4, NEON-tuned |
| MobileNetV3 FP32 (224×224) | 24 ms | 4 threads, Pi4 |
| MobileNetV3 INT8 (224×224) | 9 ms | 4 threads, Pi4 |

The NEON-tuned ORT build is important: the default ORT wheel from PyPI is compiled for a generic ARM target. For maximum Pi4 performance, build ORT from source with `-DONNX_RUNTIME_ARM_NEON_FIX=ON` or use the pre-built wheel from ORT's ARM release page.

### NVIDIA Jetson: CUDA EP without standard NVIDIA GPU tooling

On Jetson platforms (Orin, Xavier, Nano), the CUDA toolkit version may be 11.4 or earlier. ORT's official `onnxruntime-gpu` wheels require CUDA 11.8+. You have two options:

1. **JetPack-specific ORT build**: ORT maintains Jetson-compatible wheels at [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases). Download the `.whl` matching your JetPack version.

2. **TensorRT EP on Jetson**: Jetson ships TensorRT as part of JetPack. The TensorRT EP often performs better than CUDA EP on Jetson because TensorRT is tuned for Jetson's iGPU architecture (INT8 inference on Jetson AGX Orin peaks at ~275 TOPS on the DLA accelerator).

### Mobile: CoreML EP and QNN EP

For Apple Silicon iOS and macOS deployments, ORT's CoreML EP delegates computation to the Neural Engine. For Qualcomm Snapdragon-based Android devices, ORT's QNN EP (Qualcomm Neural Network) delegates to the Hexagon DSP.

Neither EP supports all ONNX ops. A common pattern is to export a simplified model (remove custom attention, use standard `MatMul` + `Add` + `GELU` instead of contrib ops) and run the full EP priority chain: `["CoreMLExecutionProvider", "CPUExecutionProvider"]`. ORT will run supported ops on Neural Engine and fall back to CPU EP for the rest.

The performance ceiling for CoreML EP on M3 MacBook Pro (base model):
- BERT-base, batch=1, seq=128: ~4 ms (vs ~45 ms on CPU EP) — **11× speedup via Neural Engine**
- DistilBERT, batch=1, seq=128: ~2 ms

## 16. Key takeaways

1. **ORT's `InferenceSession` is immutable and thread-safe.** Build N sessions at startup, pool them behind a queue, and serve concurrent requests without locks. Session pool size = ceiling(target_QPS / single_session_throughput).

2. **Optimization level matters significantly.** `ORT_ENABLE_ALL` fuses `EmbedLayerNormalization`, `SkipLayerNormalization`, and bias-GEMM into single fast kernels. Never run production inference at `ORT_DISABLE_ALL` — you are leaving 40–60% performance on the table.

3. **For BERT-style CPU inference, dynamic INT8 quantization is the default answer.** It requires zero calibration data, delivers 3–4× speedup versus FP32, compresses model size by 4×, and loses less than 0.5 F1 points on standard benchmarks. Use `per_channel=True` for better accuracy; use `reduce_range=True` on older CPUs.

4. **IOBinding is mandatory for high-throughput GPU serving.** The ~0.8 ms H→D copy tax on each inference call adds up to tens of milliseconds per second at 50+ QPS. Pre-allocate buffers, bind them once, and reuse across calls.

5. **Tune `intra_op_num_threads` explicitly.** The default (all cores) is wrong for pooled deployments. Set it to `physical_cores / num_sessions`. Verify with ORT's profiler, not intuition — what feels like "more parallelism" often becomes "lock contention at scale."

6. **Export with opset 17 and `dynamic_axes` for both batch and sequence dimensions.** Never export a fixed-shape model for production unless your serving traffic is genuinely fixed-shape. Opset 17 unlocks `LayerNormalization` fusion that saves 8–15% latency versus opset 12.

7. **ORT loses badly on LLM serving.** No continuous batching, no PagedAttention, no KV cache sharing. Use [vLLM](/blog/machine-learning/model-serving/vllm-deep-dive) or TGI for autoregressive decoding. ORT is for encoders and rerankers — the models that run once per request, not once per token.

8. **TRT EP beats CUDA EP on fixed-shape GPU workloads by 20–40%.** Build the TRT engine cache once at deployment time, not at request time. The cache is GPU-arch-specific — invalidate it on hardware upgrade, and store it in a persistent volume.

9. **Use Optimum for HuggingFace models.** `optimum-cli export onnx --optimize O2` handles the export, opset selection, and layernorm fusion in one command. `ORTModelFor*` classes are drop-in replacements for the transformers API with automatic ONNX export on first load.

10. **Benchmark with warm-up, real batch sizes, and p99 — not p50.** A 200-sample p50 is fine for median; use 500+ samples for p99. Discard the first 20 calls from every benchmark. Silent EP fallback to CPU is the most common cause of unexpected latency — always verify EP assignment via profiling before signing off on a benchmark.

## 16. Deployment hardening: versioning, rollback, and model governance

Shipping an ONNX model to production is not a one-time event. Models are retrained, quantization parameters drift as the input distribution changes, and ORT itself releases new versions with changed optimization passes. You need a reproducible path from "trained model" to "serving artifact."

### Model versioning with the ONNX file

Bake the version information into the ONNX model's metadata:

```python
import onnx
model = onnx.load("bert-large.onnx")

# Set model metadata
model.model_version = 3
meta = model.metadata_props
entry = meta.add()
entry.key = "training_run_id"
entry.value = "2026-06-15-bert-large-r3"
entry = meta.add()
entry.key = "quantization_mode"
entry.value = "dynamic_int8_per_channel"
entry = meta.add()
entry.key = "opset_version"
entry.value = "17"

onnx.save(model, "bert-large-v3.onnx")
```

At load time, your service reads these metadata fields to confirm it loaded the correct artifact:

```python
session = ort.InferenceSession("bert-large-v3.onnx")
meta = session.get_modelmeta()
print(meta.custom_metadata_map)
# {'training_run_id': '2026-06-15-bert-large-r3', ...}

assert meta.custom_metadata_map.get("quantization_mode") == "dynamic_int8_per_channel", \
    "Loaded wrong model artifact — expected dynamic INT8"
```

### Canary validation before full rollout

Before routing production traffic to a new ORT artifact, run a canary validation pass:

```python
import numpy as np

def validate_model(new_session, reference_session, validation_inputs, threshold=0.01):
    """
    Run both models on validation_inputs and compare outputs.
    Fails if mean absolute difference > threshold.
    """
    diffs = []
    for inp in validation_inputs:
        ref_out = reference_session.run(None, inp)[1]
        new_out = new_session.run(None, inp)[1]
        diffs.append(np.mean(np.abs(ref_out - new_out)))

    mean_diff = np.mean(diffs)
    if mean_diff > threshold:
        raise ValueError(
            f"Model validation failed: mean absolute output diff {mean_diff:.4f} "
            f"exceeds threshold {threshold}. Check quantization calibration."
        )
    return mean_diff
```

Run this in your CI pipeline on every new model artifact before it reaches production. A mean absolute difference of >0.01 in the embedding space often signals a quantization regression or a calibration set mismatch — catch it before it affects search quality or classification accuracy.

### ORT version pinning

ORT releases change optimization passes. A model that performs correctly on ORT 1.15 may behave differently on ORT 1.17 if a new graph optimization introduces a subtle numerical difference. Pin the ORT version in your `requirements.txt` and test upgrades explicitly:

```txt
# requirements.txt
onnxruntime==1.18.0
# or, for GPU:
onnxruntime-gpu==1.18.0
```

When upgrading ORT: export a fresh copy of the model with the new version, run your full accuracy evaluation suite, benchmark latency, and only then promote to production. A silent ORT upgrade that changes optimization behavior has caused multiple production incidents in real serving systems.

## 17. Further reading

- **ONNX Runtime documentation**: [onnxruntime.ai](https://onnxruntime.ai/docs/) — InferenceSession API, EP configuration reference, quantization guide.
- **HuggingFace Optimum**: [huggingface.co/docs/optimum](https://huggingface.co/docs/optimum/onnxruntime/overview) — `ORTModel*` classes, quantization configuration, CLI reference.
- **"ONNX Runtime: Cross-Platform, High Performance ML Inferencing"** — Jiang et al., Microsoft (2021). The original system paper describing ORT's architecture, EP plugin system, and graph optimization passes.
- **Intel Neural Compressor**: [intel.github.io/neural-compressor](https://intel.github.io/neural-compressor) — alternative INT8 quantization pipeline with PTQ and QAT support, integrates with ORT.
- Series context — start here: [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — the SLO triangle and why latency/throughput/cost are always in tension.
- Related serving runtimes: [Triton Inference Server deep-dive](/blog/machine-learning/model-serving/triton-inference-server-deep-dive) — ensemble pipelines, dynamic batching, and multi-model serving that ORT alone cannot provide.
- Choosing a serving stack: [Choosing your serving stack](/blog/machine-learning/model-serving/choosing-your-serving-stack) — the full decision matrix of Triton vs Ray Serve vs TorchServe vs vLLM, with ORT as a runtime component.
- For quantization theory: [Quantization for inference](/blog/machine-learning/model-serving/quantization-for-inference-not-training) — PTQ vs QAT, INT8/FP8/INT4 tradeoffs, and the accuracy math behind reduced-precision arithmetic.
