---
title: "The model serving stack: A map of every layer between a trained model and a user request"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand every layer of the six-layer model serving stack — from raw weights to the API gateway — so you can locate bottlenecks, choose the right framework, and reason about GPU memory before your first production incident."
tags:
  [
    "model-serving",
    "inference",
    "llm-infrastructure",
    "vllm",
    "triton-inference-server",
    "kubernetes",
    "gpu-memory",
    "ml-infrastructure",
    "serving-framework",
    "ray-serve",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-model-serving-stack-1.png"
---

It is 2:17 AM. Your new Llama-3-8B deployment has been live for six hours. Then the p99 latency alarm fires. You SSH into the serving pod, stare at `nvidia-smi`, and see GPU utilization at 12%. The GPU is almost idle, yet requests are piling up with 8-second latency. Where is the bottleneck? Is it the tokenizer? The KV cache allocator? The HTTP gateway's connection pool? The Kubernetes scheduler that's draining one node for maintenance?

You do not know, because you have not yet built a mental map of every layer that sits between your weights file and the user's browser. Without that map, debugging a production incident is guesswork. With it, you eliminate candidates in seconds.

This post builds that map. We will walk every one of the six layers in the serving stack — model artifact, inference runtime, serving framework, API gateway, infrastructure, and observability — and for each layer we will name the tools, derive the relevant math, and show you the failure modes. By the end, you will be able to read a flame graph or a `kubectl describe pod` and immediately know which layer owns the problem.

The running example is a concrete deployment: **Llama-3-8B on a single H100 80 GB GPU**, serving an OpenAI-compatible chat endpoint. We will trace one request from the client's HTTP POST all the way through to the streamed SSE response and back — every hop, every queue, every GPU kernel.

![The six-layer model serving stack, from model artifact at the bottom to observability at the top](/imgs/blogs/the-model-serving-stack-1.png)

Figure 1 above shows the full stack at a glance. The SLO triangle introduced in [what-is-model-serving](/blog/machine-learning/model-serving/what-is-model-serving) runs through every layer: latency, throughput, and cost are all negotiated simultaneously as each layer does its work.

This is the key thing that most ML practitioners get wrong when they first ship a model: they treat serving as "just running inference in a loop." It is not. It is a six-layer distributed system, and systems thinking applies. A request does not travel directly from the client to the GPU. It passes through a TLS termination proxy, a rate limiter, a request queue, a batch scheduler, a CUDA kernel dispatcher, and a KV cache allocator before the first GPU computation runs. Then on the way back it goes through a detokenizer, a streaming serializer, an HTTP response chunker, and potentially a CDN edge node before the first character reaches the client's screen.

Understanding the stack in its full depth is what separates an ML engineer who can build a demo from one who can own a production service. The demo breaks at 10 concurrent users. The production service handles 10,000 and pages you when utilization crosses 85%, not when users start complaining.

Let us build the map, layer by layer.


## Layer 1: The model artifact

The model artifact is the foundation of the entire stack. It consists of at minimum three components: the **weight tensors**, a **configuration file** describing architecture, and a **tokenizer** (for language models). Everything above this layer ultimately depends on what is encoded here.

### Artifact formats

There is no single universal format. The format you choose affects which runtimes can load the model and how much conversion overhead you pay.

**PyTorch checkpoint (`.pt` / `.bin` / safetensors)**: The raw output of training. A `state_dict` mapping parameter names to tensors. Hugging Face models ship as sharded `model.safetensors.index.json` + `model-0000X-of-0000Y.safetensors`. The safetensors format is preferred over the older pickle-based `.bin` because it prevents arbitrary code execution on load and enables zero-copy memory mapping.

**TorchScript (`.torchscript.pt`)**: A serialized, traced or scripted version of the model's `forward()` method. Does not require the original class definition at load time. Useful when you want to deploy without shipping your training codebase, but tracing fails on dynamic control flow and can silently mis-capture shapes.

**ONNX (`.onnx`)**: Open Neural Network Exchange — a graph-level intermediate representation. Framework-neutral: export from PyTorch, load in ONNX Runtime, TensorRT, OpenVINO, or CoreML. ONNX's operator set does not cover every PyTorch op, so you will encounter `torch.onnx.export` errors on custom attention kernels or router operations. The current standard is ONNX opset 17–20.

**TensorRT engine plan (`.engine` / `.plan`)**: A NVIDIA-specific serialized execution plan produced by the TensorRT builder from an ONNX graph. Hard-locked to the CUDA toolkit version, driver, and GPU architecture at build time — an engine built for H100 will refuse to load on an A100. Inference is extremely fast because TensorRT fuses operators, selects optimal CUDA kernels, and can run FP8 on Hopper.

**GGUF (`.gguf`)**: The format used by `llama.cpp` and Ollama for quantized LLMs on CPU and Apple Silicon. Stores weights with per-block or per-tensor quantization metadata, the tokenizer, and the model config in one self-describing binary. Not suitable for GPU clusters but dominant in local inference.

**SavedModel (TensorFlow)**: TensorFlow's equivalent to TorchScript — a directory with `saved_model.pb` plus variable shards. Loaded by TensorFlow Serving, TFLite, and TF-TensorRT.

For Llama-3-8B, the practical choices are:
- **safetensors** — for vLLM, TGI, or any Hugging Face-compatible server.
- **ONNX** — for ONNX Runtime or Triton's ORT backend.
- **TensorRT engine** — for Triton's TRT backend at maximum throughput.

### Format conversion pipeline

Model format conversion is a one-way, lossy-ish pipeline. You almost always start from the safetensors checkpoint and export forward:

```
safetensors → ONNX → TensorRT engine
                   → ONNX Runtime (direct)
           → GPTQ / AWQ (quantized safetensors)
           → GGUF (llama.cpp quantized)
```

Conversion for a production Llama-3-8B export to ONNX:

```bash
# Export to ONNX using optimum (Hugging Face's export tool)
pip install optimum[exporters]

optimum-cli export onnx \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --task text-generation-with-past \
  --fp16 \
  --framework pt \
  ./llama3-8b-onnx/

# Verify the export
python -c "
import onnx
model = onnx.load('./llama3-8b-onnx/model.onnx')
onnx.checker.check_model(model)
print('Model valid, opset:', model.opset_import[0].version)
"
```

The `--task text-generation-with-past` flag exports the model with KV cache inputs/outputs as explicit ONNX tensors — this is the version that can do incremental decode without recomputing all previous attention. Without this flag, every decode step recomputes the full context, making ONNX models 10–100× slower than native PyTorch for generation.

Export to TensorRT from the ONNX artifact:

```bash
# Build TensorRT engine from ONNX (this can take 20-40 minutes)
trtexec \
  --onnx=./llama3-8b-onnx/model.onnx \
  --saveEngine=./llama3-8b.engine \
  --fp16 \
  --minShapes=input_ids:1x1,attention_mask:1x1,past_key_values.0.key:1x8x0x128 \
  --optShapes=input_ids:1x512,attention_mask:1x512,past_key_values.0.key:1x8x512x128 \
  --maxShapes=input_ids:1x4096,attention_mask:1x4096,past_key_values.0.key:1x8x4096x128 \
  --verbose
```

The `--minShapes`, `--optShapes`, `--maxShapes` flags are mandatory for LLMs because token sequence length is dynamic. TensorRT builds separate calibrated kernels for the optimal shape and falls back to the min/max profiles for outliers.

### Artifact validation before serving

Never load an artifact without validating it. Common artifact corruption failures:

- **Truncated download**: the S3 copy timed out at 97% and the last shard is incomplete. The model loads (PyTorch reads what it can), but the final layers have zero-initialized weights. Output quality degrades silently — no exception is thrown.
- **Wrong quantization precision**: you thought you downloaded the BF16 model but got the INT4 GPTQ version. The model loads but outputs garbage because the runtime does not know to apply the quantization metadata.
- **Version mismatch**: tokenizer v3.0 does not understand a special token added in model v3.1. No error on load; wrong output for messages using the new token.

A minimal validation script for a safetensors checkpoint:

```python
import hashlib
import json
from pathlib import Path
from safetensors import safe_open

def validate_artifact(model_dir: str) -> bool:
    model_path = Path(model_dir)

    # 1. Check all expected files exist
    required = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    for f in required:
        if not (model_path / f).exists():
            print(f"MISSING: {f}")
            return False

    # 2. Load shard index and verify all shards are present
    with open(model_path / "model.safetensors.index.json") as fh:
        index = json.load(fh)
    shards = set(index["weight_map"].values())
    for shard in shards:
        shard_path = model_path / shard
        if not shard_path.exists():
            print(f"MISSING SHARD: {shard}")
            return False

    # 3. Try loading one shard to verify it is not corrupt
    first_shard = model_path / sorted(shards)[0]
    try:
        with safe_open(str(first_shard), framework="pt") as f:
            keys = list(f.keys())
        print(f"First shard OK: {len(keys)} tensors")
    except Exception as e:
        print(f"CORRUPT SHARD: {e}")
        return False

    # 4. Validate config can be parsed
    with open(model_path / "config.json") as fh:
        config = json.load(fh)
    print(f"Config: {config.get('model_type')}, {config.get('num_hidden_layers')} layers")

    return True

validate_artifact("./llama-3-8b/")
```

Run this check in your CI/CD pipeline before promoting a new model artifact to the serving tier. A 5-second validation catches weeks of silent degradation.

### What lives in the artifact directory

A minimal Llama-3-8B artifact on disk looks like this:

```
llama-3-8b/
├── config.json                  # model architecture hyperparameters
├── generation_config.json       # default sampling parameters
├── special_tokens_map.json      # special token IDs
├── tokenizer.json               # BPE vocabulary + merge rules
├── tokenizer_config.json        # tokenizer class + chat template
├── model.safetensors.index.json # shard manifest
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
└── model-00004-of-00004.safetensors
```

The total uncompressed size is **~16 GB in BF16** (8 billion parameters × 2 bytes). We will derive this exactly in the GPU memory section.



## Layer 2: The inference runtime

The inference runtime is the software that actually executes the model's computation. It accepts weight tensors and input data, schedules GPU kernels, and returns logits or activations.

### What a runtime does

When you call `model.forward(input_ids)` in PyTorch eager mode, CPython dispatches one Python call per operation — embedding lookup, attention, FFN, softmax, and so on. Each call has kernel launch overhead of roughly 5–20 µs. For a 32-layer transformer, that is 32 × (at minimum) 4 operations × 10 µs = 1.28 ms of pure kernel-launch overhead, before any computation. At small batch sizes, this overhead dominates.

A production runtime eliminates this overhead through three mechanisms:

1. **Operator fusion**: merge adjacent elementwise operations into a single CUDA kernel. For example, fusing the post-attention layer norm + residual add avoids two separate memory round-trips through HBM.
2. **Kernel selection**: at build time, profile multiple CUDA kernel implementations for each operation (cuBLAS GEMM strategies, tiled GEMM variants, etc.) and select the fastest for the actual tensor shapes.
3. **Graph capture**: use CUDA Graph API to record a sequence of kernel launches into a single replayable graph. Replaying the graph costs ~3 µs regardless of graph depth, versus the per-op overhead.

### Key runtimes

**PyTorch eager mode**: the default. No compilation, no fusion beyond cuDNN convolutions. Acceptable for batch sizes ≥ 32 where compute dominates overhead, but leaves 20–40% throughput on the table for LLMs.

**`torch.compile`** (PyTorch 2.0+): traces the computation graph via `torch.fx`, runs Inductor (the default backend), which generates optimized Triton kernels. For serving, use `mode="reduce-overhead"` (enables CUDA Graph) or `mode="max-autotune"` (aggressive kernel selection, longer compilation). First call triggers JIT compilation; subsequent calls use the cached compiled graph.

```python
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)
model = model.to("cuda")

# Compile for serving — reduces overhead by ~30% on H100 for bs=1
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

**ONNX Runtime (ORT)**: loads an ONNX graph and dispatches to one or more *execution providers* (EPs). The `CUDAExecutionProvider` uses cuBLAS and cuDNN. The `TensorrtExecutionProvider` builds a TRT engine on first run and caches it. ORT is the standard choice for non-LLM models in production: image classifiers, embedding models, tabular regressors.

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    "model.onnx",
    sess_options=sess_options,
    providers=[
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/tmp/trt_cache",
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)
```

**TensorRT**: NVIDIA's highest-performance runtime for GPU inference. Accepts ONNX, builds a fused, quantized execution plan. Achieves 2–4× throughput vs ORT on many vision and NLP models. The cost: 10–30 minute build times, strict version pinning, and no support for dynamic shapes without explicit profile ranges.

**FlashAttention-2**: not a runtime in itself, but a drop-in CUDA kernel for the attention operation that achieves 2–4× speedup over the standard PyTorch attention by fusing the softmax, masking, and matmul into a single kernel and never materializing the full $N \times N$ attention matrix. Available as a compiled C++ extension. vLLM, TGI, and TorchServe all use FlashAttention-2 automatically if available.

**xFormers**: Meta's library of memory-efficient operations. Provides `memory_efficient_attention`, which achieves similar memory savings to FlashAttention but is more portable across GPU generations.

The relationship between runtimes: `torch.compile` wraps PyTorch; ONNX Runtime is framework-neutral; TensorRT is ONNX Runtime's most aggressive EP; FlashAttention is a kernel drop-in within any runtime. In vLLM and TGI, FlashAttention-2 and torch.compile work together — compile handles the non-attention ops, FlashAttention handles the attention kernel.

### Benchmarking runtime choice on H100

Here is a concrete benchmark comparison for Llama-3-8B, single GPU, batch size 1 (latency-optimized), H100 80 GB SXM5:

| Runtime | TTFT (ms) | TPOT (ms) | Throughput (tok/s, bs=32) | Notes |
|---|---|---|---|---|
| PyTorch eager | 28 ms | 38 ms | 2,100 | Baseline; no optimization |
| torch.compile (reduce-overhead) | 14 ms | 28 ms | 3,400 | +62% throughput, -27% TPOT |
| torch.compile + FlashAttn-2 | 12 ms | 25 ms | 4,200 | +100% vs eager |
| ONNX Runtime + CUDA EP | 18 ms | 32 ms | 2,800 | Simpler to deploy than TRT |
| TensorRT FP16 | 9 ms | 21 ms | 5,100 | Fastest; 30-min build time |

The numbers confirm the hierarchy: eager < torch.compile < ORT < TRT. But the build complexity increases left to right. For most teams, **torch.compile + FlashAttention-2 via vLLM** hits 85–90% of TRT performance with zero additional build step. TRT is worth pursuing when you have a stable model, a dedicated infra team, and a latency SLO under 10 ms TTFT.

### The dtype-bandwidth relationship

The choice of compute dtype — BF16, FP16, FP8, INT8, INT4 — is not just about accuracy. It is about memory bandwidth. Every time the GPU loads a weight to compute a GEMM, it moves bytes from HBM to SRAM. The bandwidth is fixed at 3.35 TB/s on H100.

For a weight matrix load of 4 billion bytes (half the Llama-3-8B model in BF16):

$$t_{\text{load}} = \frac{4 \times 10^9 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} \approx 1.2 \text{ ms}$$

In FP8 (1 byte per param instead of 2), the same weight matrix loads in 0.6 ms. This is the *memory-bandwidth benefit of quantization for inference*: halving the dtype halves the load time. In the decode phase — where the GPU is nearly always memory-bandwidth-bound — this directly halves TPOT. See [quantization-for-inference-not-training](/blog/machine-learning/model-serving/quantization-for-inference-not-training) for a full treatment.



## Layer 3: The serving framework

The inference runtime knows how to execute one forward pass. The serving framework knows how to manage many concurrent forward passes efficiently: it accepts HTTP/gRPC requests, schedules them into batches, manages KV cache memory, and exposes an API.

### What the serving framework is responsible for

- **Request queueing**: hold incoming requests until a batch can be formed. Queue depth and timeout policy directly determine latency vs throughput.
- **Batching**: group multiple requests into a single batched forward pass to amortize the fixed cost of a CUDA kernel launch. More on batching algorithms in [batching-fundamentals-latency-throughput-tradeoff](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff).
- **KV cache management** (LLMs only): allocate GPU memory for attention key/value tensors per sequence, and free it when the sequence completes or is preempted.
- **API surface**: expose an OpenAI-compatible REST API, gRPC endpoints, or both.
- **Concurrency control**: prevent one slow request from starving all others; implement priority queues; handle timeouts.

### The major frameworks

**vLLM**: the de-facto standard for LLM serving. Built around *PagedAttention* — a KV cache allocator that manages GPU memory in fixed-size pages (default 16 tokens per page), analogous to virtual memory paging in an OS. This prevents KV cache fragmentation, enables sequence preemption and restart without full eviction, and achieves near-100% KV cache utilization. vLLM exposes an OpenAI-compatible API server out of the box.

**Triton Inference Server (Triton IS)**: NVIDIA's multi-framework server that spans layers 2 and 3. It manages a *model repository* (a directory structure of config + weight files), supports multiple backends (PyTorch, TensorFlow, ONNX Runtime, TensorRT, Python), implements dynamic batching at the framework level, and exposes HTTP and gRPC endpoints. The main strength is *ensemble* support: pipe several models together in a single server (preprocessor → main model → postprocessor), with Triton handling inter-model data movement.

**TorchServe**: PyTorch's official production server. A model is packaged as a `.mar` (model archive) via `torch-model-archiver`, which bundles weights + handler + Python requirements. TorchServe manages multiple model workers, implements dynamic batching, exposes management and prediction REST APIs, and emits JMX + Prometheus metrics.

**Ray Serve**: a general-purpose serving layer built on Ray actors. You annotate a class with `@serve.deployment`, give it a replica count and autoscaling config, and Ray Serve spawns it as one or more actors. The main differentiator is composition: you can pipeline multiple deployments with `DeploymentHandle`, mix CPU and GPU workers, and use Ray's existing cluster for training-serving colocation. It does not provide LLM-specific features (no KV cache management, no continuous batching).

**BentoML**: a Python-first framework that spans layers 3 and 4. You define a `@bentoml.service` class, declare hardware requirements, and `bentoml build` packages it into a Bento (a Docker-ready artifact). BentoML includes a built-in HTTP server (Starlette-based), so the gateway layer is partially handled. Useful for prototyping and for teams that want a single tool from development to deployment.

**Text Generation Inference (TGI)**: Hugging Face's LLM server. Like vLLM, it implements continuous batching and FlashAttention-2, and exposes an OpenAI-compatible API. TGI's differentiator is tight integration with the Hugging Face Hub (model download by name) and strong support for quantization (GPTQ, AWQ, EETQ) and PEFT adapters.

### A minimal vLLM server

Here is a production-ready vLLM server that handles Llama-3-8B:

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import uuid

# Configure the engine
engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,             # max context window in tokens
    gpu_memory_utilization=0.90,    # fraction of GPU HBM to use for KV cache
    tensor_parallel_size=1,         # single GPU
    enable_prefix_caching=True,     # reuse KV cache for shared prefixes
    max_num_seqs=256,               # max concurrent sequences
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(request: dict):
    messages = request["messages"]
    # Build prompt from messages using chat template
    prompt = build_prompt(messages)

    sampling_params = SamplingParams(
        temperature=request.get("temperature", 0.7),
        max_tokens=request.get("max_tokens", 512),
        stop=request.get("stop", []),
    )

    request_id = str(uuid.uuid4())

    async def generate():
        async for output in engine.generate(prompt, sampling_params, request_id):
            delta = output.outputs[0].text
            # SSE format
            yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def build_prompt(messages: list) -> str:
    # Llama-3 chat template (simplified)
    prompt = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Notice that `AsyncLLMEngine` handles the KV cache, the batching, and the generation loop. You never touch those concerns directly.

### The continuous batching algorithm inside the serving framework

The reason vLLM and TGI vastly outperform naive serving is *continuous batching* (also called iteration-level scheduling or in-flight batching). Understand it and you understand why layer 3 exists as its own layer.

In static batching — what TorchServe does by default and what you get when you roll your own serving loop — you collect N requests, run the full forward pass for all N to completion (all T tokens generated), then accept the next batch. If request A needs 10 tokens and request B needs 500 tokens, the GPU sits idle generating tokens for B while A completed long ago.

In continuous batching, the scheduler makes a scheduling decision at every *iteration* (every single token generation step). After each iteration, it checks: did any sequence in the current batch just emit an end-of-sequence token? If so, evict it from the batch and insert a waiting request. The GPU is always doing useful work, and short requests do not block long ones.

The mathematics: suppose the arrival rate is $\lambda$ requests/second, the mean service time (tokens to generate) is $\bar{S}$ tokens, and the per-token generation time is $\tau$ seconds. In static batching, the effective batch size stays fixed at N even if most requests have already completed, wasting $(1 - \rho) \times 100\%$ GPU cycles where $\rho = \lambda \bar{S} \tau N$ is the utilization. In continuous batching, utilization approaches 100% as long as the arrival rate exceeds the minimum needed to keep the batch full.

Practically, this means: on the same hardware, with the same model, continuous batching delivers 3–4× higher throughput than static batching at moderate-to-high load. This is the number one reason to use a dedicated serving framework (layer 3) rather than writing your own inference loop.

The vLLM scheduler specifically implements a *preemptive* variant: if a high-priority request arrives and the KV cache is full, it can preempt a lower-priority in-progress sequence — swapping its KV state to CPU memory — to make room. The preempted sequence resumes after higher-priority requests drain. This is PagedAttention's killer feature: preemption without discarding work.

### TorchServe: the handler-based approach

For non-LLM models, TorchServe provides a handler interface that is simpler to customize than vLLM's engine API:

```python
# custom_handler.py — a TorchServe handler for an image classifier
from ts.torch_handler.vision_handler import VisionHandler
import torch

class ClassifierHandler(VisionHandler):
    """
    Custom handler for ResNet-50 image classification.
    Inherits preprocessing/postprocessing from VisionHandler.
    """

    def initialize(self, context):
        super().initialize(context)
        # model is loaded by VisionHandler.__initialize__
        self.model.eval()
        # Warm up: run a dummy inference to trigger JIT/kernel loading
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    def postprocess(self, inference_output):
        # Return top-5 class predictions with probabilities
        probs = torch.nn.functional.softmax(inference_output, dim=1)
        top5_probs, top5_classes = torch.topk(probs, 5, dim=1)
        results = []
        for probs_batch, classes_batch in zip(top5_probs, top5_classes):
            result = {
                str(cls.item()): float(prob.item())
                for cls, prob in zip(classes_batch, probs_batch)
            }
            results.append(result)
        return results
```

```bash
# Package into a .mar archive
torch-model-archiver \
  --model-name resnet50-classifier \
  --version 1.0 \
  --serialized-file resnet50.pt \
  --handler custom_handler.py \
  --requirements-file requirements.txt \
  --export-path ./model_store

# Start TorchServe
torchserve \
  --start \
  --ncs \
  --model-store ./model_store \
  --models resnet50=resnet50-classifier.mar \
  --ts-config config.properties

# Test
curl http://localhost:8080/predictions/resnet50 \
  -T test_image.jpg
```

The `config.properties` file controls batch size, GPU allocation, and the number of worker processes:

```properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=4
job_queue_size=1000

# GPU configuration
number_of_gpu=1
default_workers_per_model=1
default_response_timeout=120

# Batch configuration
batch_size=32
max_batch_delay=100
```



## Layer 4: The API gateway and router

The serving framework handles model execution. The API gateway handles everything that is not model execution: authentication, authorization, rate limiting, routing, load balancing, request logging, and TLS termination.

### What belongs at the gateway layer

The gateway layer is a load-bearing wall, not decoration. It does five jobs that the serving framework should never do:

**TLS termination**: decrypt HTTPS at the edge so the serving pods communicate over plain HTTP within the cluster. Kubernetes Ingress controllers handle this transparently.

**Authentication and authorization**: verify API keys, JWT tokens, or OAuth2 credentials before the request reaches the serving framework. Never implement auth inside the serving framework — it couples security logic to model logic.

**Rate limiting**: prevent a single tenant from consuming all capacity. Token bucket is the standard algorithm: each tenant has a bucket of capacity $C$ that refills at rate $r$ tokens/second. A request costs $1$ token. When the bucket is empty, the gateway returns 429.

**Routing**: direct requests to the correct model or version. Canary deployments split traffic by percentage. Model routers (like NVIDIA's AIBrix or KServe) can route by model name, request content (e.g., language detection for multilingual models), or estimated compute cost.

**Request logging and tracing**: emit a trace ID per request that threads through the entire stack. OpenTelemetry is the standard; it integrates with Jaeger, Zipkin, and cloud-native distributed tracing services.

### Why NOT to put gateway logic inside vLLM

A common mistake is to add auth or routing logic directly into the Python server that wraps vLLM (the `fastapi` layer in the example above). The temptation is understandable — it saves one network hop. But it creates three problems:

1. **Scaling mismatch**: the gateway layer scales horizontally to handle connection volume; the serving layer scales to handle GPU compute. These scale at different rates. A DDoS attack sends you 100,000 unauthenticated requests per second — you want those rejected by a lightweight Nginx instance, not by your expensive GPU pod.

2. **Availability coupling**: if your auth logic has a bug and raises an exception, you want the gateway to return 500 to the client while the serving pod continues serving valid requests. Mixing them means one bad auth request can poison the serving pod's process.

3. **Separation of concerns at deploy time**: you deploy new models frequently (weekly or daily). You rotate API keys and update rate-limit policies far less often. Keeping them on separate deploy cycles reduces blast radius.

The latency cost of separating gateway from serving is one extra TCP hop within the Kubernetes cluster: roughly 0.1–0.5 ms. This is negligible against typical LLM latencies of 500 ms–15 seconds. Accept the hop.

### Envoy xDS configuration for dynamic routing

When you need to route requests to different model versions without restarting the gateway, Envoy with xDS is the solution:

```yaml
# envoy-config.yaml — static bootstrap, clusters managed via xDS
static_resources:
  listeners:
    - name: listener_0
      address:
        socket_address:
          protocol: TCP
          address: 0.0.0.0
          port_value: 10000
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: ingress_http
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: local_service
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/v1/chat"
                            headers:
                              - name: "x-model-version"
                                string_match:
                                  exact: "canary"
                          route:
                            cluster: llama3_canary
                            timeout: 120s
                        - match:
                            prefix: "/v1/chat"
                          route:
                            cluster: llama3_stable
                            timeout: 120s
                http_filters:
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
  clusters:
    - name: llama3_stable
      connect_timeout: 5s
      type: STRICT_DNS
      lb_policy: LEAST_REQUEST
      load_assignment:
        cluster_name: llama3_stable
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: vllm-stable-service
                      port_value: 8000
    - name: llama3_canary
      connect_timeout: 5s
      type: STRICT_DNS
      lb_policy: LEAST_REQUEST
      load_assignment:
        cluster_name: llama3_canary
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: vllm-canary-service
                      port_value: 8000
```

This configuration routes requests with the `x-model-version: canary` header to the canary pod and all others to stable. Switching from 5% to 100% canary traffic is a one-line config change, applied without restarting Envoy or the model pods.

### Gateway tools

**Nginx**: the ubiquitous reverse proxy. Handles TLS, load balancing, rate limiting (the `limit_req_zone` directive), and request logging. Simple, battle-tested, but requires a reload to change rate limits and lacks service-mesh features.

**Envoy**: a modern L4/L7 proxy designed for service meshes. Supports dynamic configuration via xDS APIs (no reload needed), has first-class gRPC support, and integrates with Istio. Higher configuration complexity than Nginx but far more powerful at scale.

**Kong**: an API gateway built on Nginx with a plugin ecosystem. Provides auth, rate limiting, and monitoring as first-class plugins managed via a REST admin API or declarative YAML.

**Custom FastAPI gateway**: for teams that need maximum flexibility — custom auth logic, model-aware routing, A/B test header injection — a lightweight FastAPI app sitting in front of vLLM is often the right answer. FastAPI with `httpx.AsyncClient` proxies requests in microseconds.

```python
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
import os

app = FastAPI()
VLLM_URL = os.environ.get("VLLM_BACKEND", "http://vllm-service:8000")
VALID_KEYS = set(os.environ.get("API_KEYS", "").split(","))

@app.post("/v1/chat/completions")
async def proxy_chat(
    request: dict,
    authorization: str = Header(...),
):
    # Auth check
    api_key = authorization.removeprefix("Bearer ")
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Rate limit check (simplified — use redis in production)
    # ... token bucket logic ...

    # Proxy to vLLM
    async def stream():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{VLLM_URL}/v1/chat/completions",
                json={**request, "stream": True},
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")
```

![A single chat request traced through all six layers of the serving stack](/imgs/blogs/the-model-serving-stack-2.png)

Figure 2 shows exactly how a client request traverses every layer. The gateway adds roughly 0.5–2 ms of overhead for auth and rate-limit checks — negligible compared to the 12 ms TTFT of the model, but it becomes the bottleneck if you perform a synchronous database lookup for every request.



## Layer 5: Infrastructure and orchestration

The infrastructure layer is where you run the containers that contain the serving framework. In production, this means Kubernetes with NVIDIA GPU support.

### Kubernetes for GPU serving

Kubernetes schedules pods onto nodes. For GPU workloads, the NVIDIA device plugin must be installed — it advertises `nvidia.com/gpu` as a schedulable resource. A pod requests GPU resources and Kubernetes binds it to a node with available GPUs.

A minimal Deployment for our Llama-3-8B server:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama3-8b-server
  namespace: ml-serving
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llama3-8b-server
  template:
    metadata:
      labels:
        app: llama3-8b-server
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.5.5
          args:
            - "--model"
            - "meta-llama/Meta-Llama-3-8B-Instruct"
            - "--dtype"
            - "bfloat16"
            - "--gpu-memory-utilization"
            - "0.90"
            - "--max-model-len"
            - "8192"
            - "--port"
            - "8000"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "32Gi"
              cpu: "8"
            requests:
              nvidia.com/gpu: "1"
              memory: "24Gi"
              cpu: "4"
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120   # weight loading takes time
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
      nodeSelector:
        nvidia.com/gpu.product: "H100-SXM5-80GB"
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
```

Key points:
- `resources.limits.nvidia.com/gpu: "1"` is mandatory — without it Kubernetes cannot schedule the pod onto a GPU node.
- `initialDelaySeconds: 120` on the liveness probe prevents Kubernetes from killing the pod during the 60–90 second model load time.
- The `nodeSelector` pins the pod to H100 nodes so it does not accidentally land on a T4 with 16 GB of VRAM.
- The `persistentVolumeClaim` for model cache avoids re-downloading 16 GB of weights on every pod restart.

### Kubernetes autoscaling for GPU inference

The infrastructure layer is also responsible for *scaling* — adding replicas when load increases and removing them when it drops. For GPU workloads, this is more nuanced than CPU autoscaling because:

1. GPU pods take 60–120 seconds to start (model download + load time). The Horizontal Pod Autoscaler (HPA) must be tuned to trigger scale-out before the queue is already backlogged.
2. The standard HPA metric (CPU utilization) is meaningless for GPU inference. You must scale on custom metrics: request queue depth, KV cache utilization, or RPS.

Here is an HPA backed by a custom metric from Prometheus via the Prometheus Adapter:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama3-8b-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama3-8b-server
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting   # scraped from Prometheus
        target:
          type: AverageValue
          averageValue: "10"    # scale out when avg waiting queue > 10
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60    # wait 60s before scaling up again
      policies:
        - type: Pods
          value: 2
          periodSeconds: 120    # add at most 2 replicas per 2 minutes
    scaleDown:
      stabilizationWindowSeconds: 300   # wait 5 minutes before scaling down
      policies:
        - type: Pods
          value: 1
          periodSeconds: 300
```

The `stabilizationWindowSeconds` settings are critical for LLMs. Without them, the HPA might add 4 pods at once (triggering 4 × 2-minute model loads, temporarily making things worse) or remove pods while requests are still in flight. The conservative behavior block above ensures scaling happens gradually.

For scale-to-zero (removing the last replica during idle periods), use KEDA (Kubernetes Event-Driven Autoscaling), which the standard HPA cannot do (minimum replicas = 1). KEDA's ScaledObject can check a Prometheus metric and scale to zero after a configurable idle period.

### Container images for GPU serving

The starting point is always an NVIDIA base image:

```bash
# vLLM pre-built image — recommended for production
docker pull vllm/vllm-openai:v0.5.5

# Or build from NVIDIA's CUDA base if you need a custom runtime
docker build -t my-vllm:latest - <<'EOF'
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Python dependencies
RUN pip install vllm==0.5.5 flashattn>=2.5.0 transformers>=4.40.0

# Copy application
COPY server.py /app/server.py
WORKDIR /app

CMD ["python", "server.py"]
EOF

# Run with GPU access
docker run --gpus all -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  my-vllm:latest
```

Always pin the CUDA version in the base image. A mismatch between the CUDA version in the container and the driver on the host node causes cryptic `libcuda.so` errors. For H100, you need CUDA 12.1+ and driver ≥ 530.

### Triton model repository layout

For Triton Inference Server, the model is not loaded by path — it is served from a structured repository:

```
model_repository/
├── llama3_ensemble/
│   ├── config.pbtxt              # ensemble pipeline config
│   └── 1/                        # version directory
├── llama3_tokenize/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py              # Python backend for tokenizer
├── llama3_trt/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan            # TensorRT engine
└── llama3_detokenize/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

```bash
# config.pbtxt for the TRT model backend
name: "llama3_trt"
backend: "tensorrt"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]    # dynamic sequence length
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP16
    dims: [-1, 128256]   # vocab size
  }
]

dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16, 32]
  max_queue_delay_microseconds: 1000   # wait up to 1 ms to fill batch
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```

```bash
# Start Triton with the model repository
tritonserver \
  --model-repository=/models \
  --backend-config=tensorrt,coalesce-request-input=true \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --log-verbose=1
```

![The framework-to-stack-layer coverage matrix: which layers each framework owns](/imgs/blogs/the-model-serving-stack-3.png)

Figure 3 is the coverage matrix. The critical insight: **Triton and vLLM both cover layers 2 and 3**, but they are not substitutes. Triton is general-purpose — it handles vision models, tabular models, and multi-model ensembles beautifully. vLLM is LLM-specific — it implements PagedAttention, prefix caching, and speculative decoding that Triton does not have. Choose Triton when you need a multi-model pipeline or non-LLM workloads; choose vLLM when you are serving LLMs at scale.



## Layer 6: Observability

The observability layer is often bolted on as an afterthought. That is a mistake. You cannot debug a production issue without metrics, traces, and logs. More specifically, you cannot do **layer-aware debugging** without metrics tagged by layer.

### The three pillars: metrics, traces, and logs

Observability is not metrics alone. The three pillars serve different purposes:

**Metrics** answer "what is the system doing right now?" They are aggregated, low-cardinality numbers: RPS, p99 latency, GPU utilization percentage, KV cache usage. Prometheus scrapes and stores these as time-series. A well-designed dashboard answers "is the system healthy?" in three seconds.

**Traces** answer "what happened to this specific request?" A distributed trace is a tree of spans, each representing one operation (auth check, tokenize, prefill pass, decode iteration, SSE flush). When a specific user reports their request took 8 seconds, you pull their trace and see which span took 7.5 seconds. Without traces, you have to reproduce the issue, which is often impossible in production.

**Logs** answer "what exactly happened, and in what order?" Structured logs (JSON) are far more useful than unstructured text. Every log line should carry `trace_id`, `model_version`, `request_id`, and `duration_ms` as fields. This lets you join logs to traces and filter by request ID in sub-second time.

The golden rule: **any alert that fires should be answerable by a metric**. Any metric anomaly should be diagnosable by a trace. Any trace anomaly should have supporting log context. If any of these links is broken, you are flying partially blind.

### Metrics to collect

At minimum, collect these per-layer metrics:

| Layer | Metric | Why it matters |
|---|---|---|
| Model artifact | none at runtime | Validate on load |
| Runtime | GPU utilization %, HBM bandwidth GB/s | Identifies compute vs memory bottleneck |
| Serving framework | queue depth, batch size, KV cache usage %, tokens/s | LLM throughput health |
| Serving framework | TTFT ms (p50/p95/p99), TPOT ms | Latency SLOs |
| API gateway | RPS, 4xx rate, 5xx rate, auth latency ms | User-facing health |
| Infrastructure | pod CPU %, node GPU %, OOM kills | Scaling signals |
| Observability | scrape success rate | Meta-health |

vLLM exposes Prometheus metrics at `/metrics` automatically. The key ones:

```yaml
# Prometheus scrape config for vLLM
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["vllm-service:8000"]
    metrics_path: /metrics
    scrape_interval: 15s
```

Useful Prometheus queries:

```promql
# p99 TTFT in seconds
histogram_quantile(0.99,
  rate(vllm:time_to_first_token_seconds_bucket[5m])
)

# KV cache utilization (> 0.9 means memory pressure)
vllm:gpu_cache_usage_perc

# Tokens per second (throughput)
rate(vllm:generation_tokens_total[1m])

# Request queue depth
vllm:num_requests_waiting
```

### OpenTelemetry tracing

Distributed tracing lets you follow a single request across the gateway, serving framework, and runtime. The standard is OpenTelemetry (OTEL).

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure OTLP exporter (e.g., to Jaeger)
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)
trace.set_tracer_provider(provider)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

tracer = trace.get_tracer("serving-gateway")

@app.post("/v1/chat/completions")
async def chat(request: dict):
    with tracer.start_as_current_span("chat_completion") as span:
        span.set_attribute("model", request.get("model", "unknown"))
        span.set_attribute("max_tokens", request.get("max_tokens", 0))
        # ... proxy to vLLM ...
```

A properly instrumented trace shows you: how much time the gateway spent on auth (should be < 1 ms), how long the request queued in vLLM (queue depth metric), how long prefill took (TTFT), and how long decode ran (total time − TTFT). When p99 latency spikes, you read the trace, not the logs.

### Grafana dashboard configuration

A minimal but useful Grafana dashboard for an LLM serving deployment queries four panels:

```promql
# Panel 1: Request rate (RPS)
sum(rate(vllm:request_success_total[1m])) + sum(rate(vllm:request_failure_total[1m]))

# Panel 2: p50/p95/p99 TTFT
histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket[5m]))
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[5m]))
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# Panel 3: KV cache utilization (alert threshold: 0.85)
vllm:gpu_cache_usage_perc

# Panel 4: Waiting requests (alert threshold: 50)
vllm:num_requests_waiting
```

Set alert rules for:
- `vllm:gpu_cache_usage_perc > 0.85` for 2 minutes → PagerDuty (imminent OOM)
- `vllm:num_requests_waiting > 100` for 5 minutes → Slack (queue building, add capacity)
- `histogram_quantile(0.99, ...) > 0.050` for 3 minutes → PagerDuty (p99 TTFT SLO breach)

The observability layer does not just catch incidents — it also provides the data for capacity planning. A weekly review of `max(vllm:gpu_cache_usage_perc)` and `max(vllm:num_requests_waiting)` tells you whether you need to provision another GPU before traffic grows, not after the first OOM at 3 AM.



## The GPU memory budget: A derived formula

Before you deploy a model, you must know whether it fits in GPU memory — not just the weights, but everything at serving time. The memory budget equation:

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{kvcache}} + M_{\text{activations}} + M_{\text{overhead}}$$

**Model weights:**

$$M_{\text{weights}} = P \times B_{\text{param}}$$

where $P$ is the number of parameters and $B_{\text{param}}$ is bytes per parameter. For Llama-3-8B in BF16:

$$M_{\text{weights}} = 8 \times 10^9 \times 2 \text{ bytes} = 16 \text{ GB}$$

For FP8 (H100 native), this halves to 8 GB. For INT4 (GPTQ), it drops to 4 GB.

**KV cache:**

The KV cache stores key and value tensors for every layer and every token in every active sequence. For Llama-3-8B: 32 layers, hidden dimension 4096, grouped-query attention with 8 KV heads (each head dim = 128):

$$M_{\text{kvcache}} = L \times 2 \times S \times H_{\text{kv}} \times d_{\text{head}} \times N_{\text{seqs}} \times B_{\text{kv}}$$

where:
- $L = 32$ (transformer layers)
- $2$ (key and value)
- $S$ = sequence length in tokens
- $H_{\text{kv}} = 8$ (KV heads in GQA)
- $d_{\text{head}} = 128$ (head dimension)
- $N_{\text{seqs}}$ = number of concurrent sequences
- $B_{\text{kv}} = 2$ bytes (BF16)

For $S = 2048$ tokens and $N_{\text{seqs}} = 256$:

$$M_{\text{kvcache}} = 32 \times 2 \times 2048 \times 8 \times 128 \times 256 \times 2$$

$$= 32 \times 2 \times 2048 \times 8 \times 128 \times 256 \times 2 = 34{,}359{,}738{,}368 \text{ bytes} \approx 32 \text{ GB}$$

So weights (16 GB) + KV cache (32 GB) = 48 GB. On an H100 80 GB, that leaves ~30 GB for activations and overhead. Comfortable.

On an A100 40 GB, you have a problem: weights alone are 16 GB, so you have only 24 GB for KV cache — limiting you to roughly 128 concurrent sequences at 2048 tokens. This is why vLLM's `gpu_memory_utilization` parameter is so important: setting it to 0.90 on an H100 allocates 72 GB, reserving 8 GB for activations and CUDA overhead.

### Memory budget across GPU tiers

Here is the practical memory budget for Llama-3-8B across common GPU choices, at 2048-token max context:

| GPU | VRAM | Weights (BF16) | Available KV pool | Max concurrent seqs | Notes |
|---|---|---|---|---|---|
| T4 16 GB | 16 GB | 16 GB | 0 GB | 0 | Cannot fit BF16; use INT8 (8 GB) → 6 GB KV |
| A10G 24 GB | 24 GB | 16 GB | ~6 GB | ~45 | Tight; use INT8 for more headroom |
| A100 40 GB | 40 GB | 16 GB | ~22 GB | ~165 | Comfortable for dev; production needs 80 GB |
| A100 80 GB | 80 GB | 16 GB | ~60 GB | ~450 | Good production tier |
| H100 80 GB | 80 GB | 16 GB | ~60 GB | ~450 | Same math; H100 is faster, not bigger |
| H100 NVL 94 GB | 94 GB | 16 GB | ~74 GB | ~550 | Large KV pool; great for long contexts |

For the T4, which has only 16 GB, you cannot serve Llama-3-8B in BF16 at all — the weights alone fill the GPU. The options are: (1) quantize to INT8 (8 GB weights, 6 GB KV, ~45 sequences); (2) use a smaller model (Llama-3-2B fits easily); or (3) use CPU offloading (slow, but functional for dev). See [quantization-for-llm-serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) for the memory-accuracy trade-off analysis.

### The memory bandwidth bottleneck in the decode phase

Memory equations also explain *why* the decode phase is slow. During decode, at each step we:
1. Load all model weights from HBM: $16 \text{ GB} / 3.35 \text{ TB/s} = 4.8 \text{ ms}$
2. Load the KV cache for the current batch's sequences.
3. Run a tiny GEMM (batch_size × 1 × hidden_dim, effectively a matrix-vector product).

At batch size 1, the compute time for step 3 is negligible — the GPU finishes the GEMM in microseconds. The 4.8 ms is the floor imposed by loading the weights. TPOT of 25 ms (from our earlier benchmark) breaks down as: 4.8 ms weights load + 1.2 ms KV load + ~19 ms other overhead (kernel launch, scheduling, memory allocation). This is why H100 (3.35 TB/s HBM) is faster than A100 (2.0 TB/s HBM) for decode — not because of FLOPS, but because of memory bandwidth.

The formula for the theoretical TPOT floor:

$$\text{TPOT}_{\text{min}} = \frac{M_{\text{weights}} + M_{\text{kvcache\_batch}}}{B_{\text{HBM}}}$$

For H100, single sequence, Llama-3-8B: $(16 + 0.134) \text{ GB} / 3.35 \text{ TB/s} \approx 4.8 \text{ ms}$.
For A100, same: $(16 + 0.134) \text{ GB} / 2.0 \text{ TB/s} \approx 8.1 \text{ ms}$.

Actual measured TPOT is higher than this floor because of kernel launch overhead, memory allocation, and Python interpreter overhead. But the floor tells you the target: if your measured TPOT is 3× the bandwidth floor, there is room to optimize. If it is 1.2× the floor, you are already close to hardware limits.

**Activations:**

Activation memory during inference (not training — no gradients) is bounded by the largest intermediate tensor. For a 32-layer model with max batch size $B$ and sequence length $S$, the peak activation buffer is roughly:

$$M_{\text{activations}} \approx B \times S \times d_{\text{model}} \times 2 \text{ bytes} \times \text{pipeline\_depth\_factor}$$

For our 8B model ($d_{\text{model}} = 4096$), $B = 32$, $S = 2048$:

$$M_{\text{activations}} \approx 32 \times 2048 \times 4096 \times 2 \approx 0.5 \text{ GB}$$

Activations are small relative to weights and KV cache for typical serving batch sizes.

![GPU memory layout for Llama-3-8B on H100 80 GB](/imgs/blogs/the-model-serving-stack-4.png)

Figure 4 shows the memory stack. The KV cache pool is the largest and most dynamic component — it grows with concurrency and sequence length, and it is the first thing you tune when you hit OOM.



## Architectural patterns: How layers compose

Real deployments combine the layers in several patterns. Understanding these patterns helps you design for your scale and constraints.

### Pattern 1: Monolithic serving (all layers in one pod)

The simplest pattern: vLLM runs inside a Docker container, Nginx runs as a sidecar or in the same pod, Prometheus scrapes the pod directly, and Kubernetes manages lifecycle. Everything in one Kubernetes Deployment.

**When to use it**: single-model deployments, teams with < 50 QPS, prototypes going to production. Simple to reason about, simple to debug.

**When not to use it**: when you need to scale the gateway independently of the model server (e.g., auth is rate-limiting but the GPU has spare capacity), or when you run multiple models that need shared gateway resources.

### Pattern 2: Sidecar pattern

The gateway runs as a Kubernetes sidecar container in the same pod as the serving framework. The sidecar handles auth and rate limiting; the main container handles inference. They communicate over localhost, so there is no network hop.

```yaml
spec:
  containers:
    - name: vllm          # main container
      image: vllm/vllm-openai:v0.5.5
      ports:
        - containerPort: 8000
    - name: auth-proxy    # sidecar
      image: my-auth-proxy:latest
      ports:
        - containerPort: 9000   # external-facing
      env:
        - name: UPSTREAM
          value: "localhost:8000"
```

### Pattern 2b: Multi-replica with least-outstanding-requests load balancing

For high-availability deployments where you run multiple vLLM replicas (e.g., 3 pods, each on one H100), you need a load balancer that is LLM-aware. Standard round-robin is wrong for LLMs: it will send a 10-token request and a 10,000-token request to the same pod with equal weight, causing the pod serving the long request to accumulate queue depth while others sit idle.

The correct algorithm is **least-outstanding-requests (LOR)**: route each new request to the replica with the fewest requests currently in flight. This naturally handles heterogeneous request lengths because long requests keep a pod busy, reducing its LOR score, while short requests quickly complete and free the pod.

```python
# Simplified LOR load balancer (stateful, in-process)
from dataclasses import dataclass
import asyncio
import httpx

@dataclass
class Backend:
    url: str
    outstanding: int = 0

class LORLoadBalancer:
    def __init__(self, backends: list[str]):
        self.backends = [Backend(url=url) for url in backends]
        self._lock = asyncio.Lock()

    async def pick(self) -> Backend:
        async with self._lock:
            return min(self.backends, key=lambda b: b.outstanding)

    async def proxy(self, request: dict) -> bytes:
        backend = await self.pick()
        async with self._lock:
            backend.outstanding += 1
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{backend.url}/v1/chat/completions",
                    json=request,
                )
            return response.content
        finally:
            async with self._lock:
                backend.outstanding -= 1
```

This is a simplified version of what AIBrix and KServe implement in their inference routers. Production implementations add health checking, circuit breaking, and retry with backoff.

### Pattern 3: Disaggregated serving (PD split)

For high-throughput LLM serving at scale, *prefill* (computing KV from the prompt) and *decode* (generating tokens one at a time) are separated into different worker pools. Prefill is compute-bound; decode is memory-bandwidth-bound. Running them on the same GPU means compute resources sit idle during decode.

In disaggregated serving:
- **Prefill workers**: receive the full prompt, run one forward pass to compute all KV states, then transfer the KV tensor to a decode worker via NCCL P2P.
- **Decode workers**: receive the pre-computed KV state and run the auto-regressive decode loop.

The KV transfer latency is roughly:

$$t_{\text{transfer}} = \frac{M_{\text{kvcache\_per\_seq}}}{B_{\text{NVLink}}}$$

For Llama-3-8B, one sequence at 2048 tokens has a KV cache of roughly $32 \times 2 \times 2048 \times 8 \times 128 \times 2 = 134 \text{ MB}$. Over NVLink at 600 GB/s: $134 \text{ MB} / 600 \text{ GB/s} \approx 0.22 \text{ ms}$. This is acceptable overhead for a long response.

vLLM v0.5+ supports PD disaggregation via the `--preemption-mode` and experimental disaggregated serving flags. Tencent and Xiaomi published production results showing 2× throughput gains from this approach — see the H1 post on [prefill-decode-disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) for the full analysis.

![Monolithic versus disaggregated serving: latency and throughput comparison](/imgs/blogs/the-model-serving-stack-5.png)

Figure 5 shows the before-after. The key cost of disaggregation is the NCCL P2P transfer between prefill and decode workers. For short sequences (< 256 tokens), this overhead is not worth it. For long contexts (> 2048 tokens), disaggregation is the right call.



## Framework selection: Which layer does each framework own?

The confusion in the ecosystem is that "vLLM", "Triton", and "Ray Serve" are often compared as if they are substitutes. They are not. Let's be precise about what each framework replaces.

### Framework decision branches

![Framework selection decision: LLM workload routes to vLLM, pipeline to Triton, general async to Ray Serve](/imgs/blogs/the-model-serving-stack-6.png)

Figure 6 is the decision tree. Work through it for your use case:

**vLLM** replaces layers 2 and 3 for LLMs:
- Layer 2: uses FlashAttention-2 kernels and `torch.compile` automatically.
- Layer 3: implements PagedAttention KV cache, continuous batching, prefix caching.
- Does NOT handle: TLS, auth, rate limiting (layer 4), scheduling (layer 5), metrics scraping (layer 6). You add those separately.

**Triton Inference Server** replaces layers 2 and 3 for general models:
- Layer 2: selects between TensorRT, ORT, PyTorch, and Python backends.
- Layer 3: dynamic batching, ensemble, gRPC/HTTP endpoints.
- Does NOT handle: auth, rate limiting (layer 4), Kubernetes scheduling (layer 5). You add Nginx or Envoy in front.

**Ray Serve** replaces only layer 3:
- Layer 2: you bring your own runtime (PyTorch eager, ORT, etc.).
- Layer 3: actor-based request routing, autoscaling, composition.
- Optionally handles parts of layer 4 via FastAPI route decorators.

**BentoML** replaces layers 3 and part of layer 4:
- Built-in HTTP server (Starlette) handles basic routing.
- Does NOT implement LLM-specific KV cache management.

**TGI (Text Generation Inference)** replaces layers 2 and 3 for LLMs, similar to vLLM:
- Uses FlashAttention-2 and custom Rust-based tokenizer.
- Stronger quantization support (GPTQ, AWQ, EETQ) out of the box.
- Preferred over vLLM when you are already deep in the Hugging Face ecosystem.



## Worked examples

#### Worked example: Sizing an H100 deployment for 100 QPS

**Goal**: serve 100 requests per second, each with a 256-token prompt and 512-token response, with p99 TTFT < 50 ms and p99 TPOT < 30 ms.

**Step 1: Compute throughput requirement.**

Each request generates 512 output tokens. At 100 QPS, peak output throughput = 51,200 tokens/s. On H100 80 GB with vLLM + continuous batching, Llama-3-8B achieves roughly **8,000–12,000 tokens/s** (from vLLM benchmarks, H100, BF16, batch size ≈ 32). We need more than one GPU.

With 2 H100s (tensor-parallel-size=2), throughput scales to ~18,000–22,000 tokens/s. At 51,200 tokens/s requirement, we need roughly **3 × (2-GPU) serving pods** = 6 H100s, leaving 30% headroom for bursts.

**Step 2: Memory budget per pod.**

- Model weights (BF16, 2-GPU TP): $16 \text{ GB} / 2 = 8$ GB per GPU.
- KV cache per GPU at `gpu_memory_utilization=0.90`: $80 \times 0.90 - 8 - 2 = 62$ GB per GPU.
- Max concurrent sequences per GPU (at 2048 tokens): $62 \times 10^9 / (32 \times 2 \times 2048 \times 4 \times 128 \times 2) = 62 \text{ GB} / 0.134 \text{ GB per seq} \approx 463$ sequences.

Very comfortable. You could serve 100 QPS with the KV cache at < 25% utilization.

**Step 3: Gateway sizing.**

Each pod has an H100 and runs the vLLM engine. In front, a Kong gateway on 2 CPU pods handles auth + rate limiting. Target: < 1 ms gateway overhead (auth from in-memory token list, no DB lookup).

**Total cost estimate (H100 on-demand cloud, ~\$3.50/hr per GPU)**:
- 6 H100 GPUs × \$3.50/hr = \$21/hr = \$504/day.
- Cost per 1M tokens: at 51,200 tok/s × 3600 s/hr = 184M tok/hr production rate.
- \$21/hr / 184M tok/hr ≈ **\$0.11 per 1M tokens** at 100% utilization. Real utilization is 50–70%, so effective cost is \$0.16–\$0.22/1M tokens.

#### Worked example: Cost modeling for a Llama-3-8B API service

**Scenario**: You are building a B2B API service that charges customers per 1,000 output tokens. You need to price it correctly to be profitable while remaining competitive.

**Step 1: Measure your cost basis.**

Running one H100 on-demand at a major cloud provider costs approximately \$3.50–\$5.00/hr. For this analysis, use \$4.00/hr.

Measured throughput (from benchmark): at steady-state serving mix (256 tokens in, 512 tokens out, 50 req/s), an H100 running vLLM achieves 12,000 output tokens/second.

**Step 2: Compute cost per token.**

$$ \text{cost per token} = \frac{\$4.00/\text{hr}}{12{,}000 \text{ tok/s} \times 3600 \text{ s/hr}} = \frac{\$4.00}{43{,}200{,}000} \approx \$0.000093 \text{ per token} $$

That is **\$0.093 per 1,000 tokens** at 100% utilization.

**Step 3: Account for actual utilization.**

Production GPU utilization averages 50–70% (traffic is not flat). At 60% utilization:

$$ \text{effective cost} = \frac{\$0.093}{0.60} \approx \$0.155 \text{ per 1k output tokens} $$

**Step 4: Add infrastructure overhead.**

Gateway pods (2 × 2 vCPU): ~\$0.20/hr. Prometheus + Grafana: ~\$0.10/hr. Network egress (streaming responses): ~\$0.05/hr. Total overhead: \$0.35/hr = \$0.35 / (0.6 × 43.2M tok/hr × 0.001) ≈ \$0.014 per 1k tokens.

**Step 5: Set a price.**

Raw cost: \$0.155 + \$0.014 = \$0.169 per 1k output tokens.
With 50% gross margin: **price = \$0.34 per 1k output tokens**.
With 70% gross margin (enterprise tier): **\$0.56 per 1k output tokens**.

For reference, OpenAI's GPT-4o-mini output pricing is \$0.60 per 1M output tokens = \$0.0006 per 1k. At \$0.34 per 1k, you are 560× more expensive than a frontier API — which only makes sense if you offer data-privacy guarantees, custom fine-tuning, on-premises deployment, or specialized domain performance. This analysis is why most teams using a commodity open-weight model (Llama-3-8B) do not build a general-purpose API; they run it for internal use where the value is privacy and cost control, not reselling tokens.

#### Worked example: Diagnosing a 2 AM latency spike

**Symptom**: p99 TTFT spikes from 18 ms to 4,200 ms. GPU utilization drops from 78% to 11%.

**Layer-by-layer diagnosis:**

1. **Observability layer first**: check Prometheus dashboard. `vllm:num_requests_waiting` is at 842 (normally < 10). Queue depth spike = serving framework overwhelmed.

2. **Serving framework**: check `vllm:gpu_cache_usage_perc` = 0.97 (97% full). KV cache is nearly exhausted. vLLM is preempting sequences — saving their KV state to CPU memory and reloading — causing the GPU to stall.

3. **Root cause**: a new API consumer started sending requests with `max_tokens=8192` (8k token responses). Each such request holds 8192 tokens of KV cache per layer for its full lifetime — roughly $32 \times 2 \times 8192 \times 8 \times 128 \times 2 = 536 \text{ MB}$ per sequence. Ten concurrent 8k requests exhaust the available KV pool.

4. **Fix**: add a per-tenant `max_tokens` cap at the gateway layer (max 2048 for default tier). Add a KV cache utilization alert at 80%. Expose a `max_model_len=4096` flag in the vLLM EngineArgs to hard-cap sequence length.

**Time to diagnosis with proper observability**: under 3 minutes. Without it: 45+ minutes of SSH and `nvidia-smi` guessing.



## Case studies and benchmarks

### Benchmarking methodology: why naive benchmarks lie

Before diving into the case studies, a note on methodology. Most LLM serving benchmarks are reported at a *fixed concurrency* (e.g., "32 concurrent users") or *fixed request rate* (e.g., "50 requests per second"). Neither is how production traffic actually behaves.

Production traffic is Poisson-distributed with bursts. The correct benchmark measures *request rate vs p99 latency* as a curve, not a single point. The curve reveals the *knee* — the request rate at which p99 latency starts climbing super-linearly. The knee is your practical capacity ceiling. Operating above the knee means small traffic increases cause large latency degradations.

For Llama-3-8B on H100, the knee typically appears at 60–80% of the peak measured throughput. A system that achieves 100 req/s under optimal conditions should be provisioned to handle 70 req/s in production, leaving 30% headroom for bursty traffic.

To generate a proper throughput-latency curve, use vLLM's built-in benchmark tool:

```bash
# Install benchmark dependencies
pip install vllm[benchmark]

# Run benchmark with Poisson arrivals
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 &

# Wait for server to load (~60s)
sleep 90

# Sweep request rates from 10 to 100 req/s
for RATE in 10 20 30 40 50 60 70 80 90 100; do
  python benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://localhost:8000 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate $RATE \
    --num-prompts 500 \
    --save-result \
    --result-dir ./results \
    --result-filename "rate_${RATE}.json"
done
```

The output for each rate gives you mean TTFT, mean TPOT, p99 TTFT, p99 TPOT, and successful request rate. Plot p99 TTFT vs request rate to find the knee. This is the benchmark that matters for production sizing.

### vLLM paper benchmarks (Kwon et al., 2023)

The original vLLM paper ("Efficient Memory Management for Large Language Model Serving with PagedAttention") benchmarks Llama-13B on A100 40 GB against a static-batching baseline (similar to TorchServe or a naive Triton backend). Results:

- **Throughput**: vLLM achieves 2–4× higher throughput than the static-batching baseline at the same mean request latency.
- **KV cache fragmentation**: the static baseline wastes 40–60% of KV cache memory due to over-allocation (reserving max_seq_len for every request); PagedAttention drops this to < 5%.
- **Memory-constrained regimes**: on A100 40 GB with Llama-13B (26 GB weights in FP16), the static baseline can serve at most 5–6 concurrent sequences; vLLM serves 20–30 with equal or better latency.

### TGI vs vLLM on H100 (Hugging Face engineering blog, 2024)

Hugging Face's engineering team benchmarked TGI v2.0 and vLLM v0.4.0 on Llama-3-70B using two H100 SXM5 GPUs (tensor parallel 2). At 100 concurrent users:

- TGI mean TTFT: 82 ms; vLLM: 78 ms (within margin of error).
- TGI mean TPOT: 19 ms; vLLM: 22 ms (TGI slightly better per-token).
- TGI peak throughput: 3,840 tok/s; vLLM: 4,100 tok/s.

Takeaway: both frameworks perform similarly at this scale. TGI's advantage is HuggingFace Hub integration; vLLM's advantage is faster iteration on features like prefix caching and speculative decoding.

### Triton TensorRT backend on ResNet-50 (NVIDIA Triton benchmarks)

For non-LLM vision models, Triton with the TensorRT backend dominates:

- ResNet-50, A100, batch size 32, FP16: **11,200 images/s** with TRT backend vs 6,800 images/s with PyTorch eager backend (+65%).
- p99 latency: 4.2 ms (TRT) vs 6.9 ms (PyTorch eager).
- GPU utilization: 96% (TRT) vs 71% (PyTorch eager).

This is the canonical argument for using Triton + TRT for computer vision serving.



## When to use each pattern (and when not to)

### vLLM — use it when:

- You are serving an open-weight LLM (Llama, Mistral, Qwen, Phi) at any scale.
- You need streaming token output with SSE.
- You care about per-token latency (TPOT) at high concurrency.
- You want prefix caching to reduce redundant KV computation (system prompts, RAG prefixes).

**Do NOT use vLLM when**: you need multi-model ensembles (vLLM serves one model per process), your model is not a decoder-only transformer, or you are on CPU.

### Triton IS — use it when:

- You have a multi-model pipeline (tokenizer → encoder → classifier) and want to manage it as a single server.
- Your model is a CNN, tabular model, or encoder that is not LLM-specific.
- You want TensorRT acceleration and the model compiles cleanly.
- You need gRPC for high-throughput server-to-server calls.

**Do NOT use Triton when**: you need LLM-specific features (continuous batching, KV cache management), your team cannot invest in the model repository setup, or you are in pure Python development mode.

### Ray Serve — use it when:

- You have complex serving logic (multi-model chains, A/B test routing, custom pre/postprocessing in Python).
- You are already using Ray for training or hyperparameter tuning and want a unified cluster.
- You need fine-grained autoscaling at the model level (different scale-up policies per model).

**Do NOT use Ray Serve when**: you need LLM-specific KV management, or your team is not comfortable with Ray's actor model and debugging tools.

### BentoML — use it when:

- You are a small team that wants a single tool from experimentation to production.
- You need a quick path from a Jupyter notebook to a deployable container.
- You are not yet at a scale where gateway-layer concerns (auth, rate limiting) need to be separate.

**Do NOT use BentoML when**: you need high-throughput LLM serving with KV cache optimization, or you need enterprise-grade gateway features.



## The request lifecycle in full

Let us trace one Llama-3-8B request all the way through, with real timing numbers for H100.

![Request lifecycle for a Llama-3-8B chat turn on H100, from client to first token](/imgs/blogs/the-model-serving-stack-7.png)

Figure 7 is the timeline:

1. **t=0 ms**: Client sends `POST /v1/chat/completions` with `{"messages": [...], "max_tokens": 512}` over HTTPS.
2. **t=0.5 ms**: Kong gateway decrypts TLS (handled by the Ingress controller, not Kong itself), validates the API key against an in-memory token set. Adds trace ID header. Forwards to vLLM service via internal HTTP.
3. **t=1 ms**: vLLM tokenizer processes the message list through the Llama-3 chat template, producing 256 token IDs. The request enters the `AsyncScheduler` queue.
4. **t=1–12 ms**: The scheduler assigns the request to the next available batch slot. The prefill pass runs: 256 tokens × 32 layers × attention + FFN = ~10 ms on H100. FlashAttention-2 computes QKV projections and attention in a single fused kernel. The KV tensors are written to the PagedAttention block table (16 pages × 16 tokens = 256 slots allocated). Logits for the last token are sampled to produce token 257.
5. **t=12 ms**: **First token emitted** (this is the TTFT). The SSE response begins streaming to the client.
6. **t=12–12,800 ms**: Decode loop. Each iteration: one forward pass over 1 token (just looking up position 257, 258, … in KV cache). H100 memory bandwidth 3.35 TB/s, KV cache read per token per layer = $8 \times 128 \times 2 \text{ bytes} = 2048$ bytes, for 32 layers = 65,536 bytes. Time to read = 65,536 / (3.35 × 10^12) ≈ 0.02 ms. But we also run FFN and projection: roughly 25 ms per token at batch size 1 in BF16. That is the TPOT.
7. **t≈12,800 ms**: The 512th token is a special `<|eot_id|>` (end-of-text). vLLM closes the generation loop, frees the KV cache pages, and sends `data: [DONE]\n\n` over SSE.

Total wall time: ~12.8 seconds for a 512-token response. This is compute-bound on the decode side (25 ms/token × 512 = 12.8 s). At batch size 32 (32 concurrent requests all in the decode phase simultaneously), the GPU runs 32 tokens in parallel for the same per-token wall time, yielding 32× throughput = ~1,280 tokens/s total.



## Common failure modes, by layer

Every layer has a canonical failure mode. Misattributing a symptom to the wrong layer is the number one cause of long-running production incidents.

![Stack-layer failure modes: each layer's failure, symptom, and fix](/imgs/blogs/the-model-serving-stack-8.png)

Figure 8 is the failure matrix. Memorize the column headings: Failure Mode → Symptom → Fix. The critical insight: the same *symptom* — high p99 latency — has different root causes depending on which layer is failing.

- **Model artifact**: a corrupt checkpoint or wrong dtype produces NaN outputs or degraded accuracy. The symptom is NOT latency — it is quality metrics declining silently. Validate weights on load by computing a checksum and doing a small reference inference.
- **Inference runtime**: if no CUDA kernel exists for the operation on the target hardware (e.g., FlashAttention-2 not compiled for the specific CUDA version), PyTorch falls back to CPU. CPU utilization spikes, GPU utilization drops to near zero. Fix: update CUDA toolkit, or explicitly install the correct wheel.
- **Serving framework**: KV cache OOM is the most common LLM-specific failure. Symptoms: `CUDA out of memory` errors in logs, vLLM's retry counter climbing. Fix: reduce `max_num_seqs`, cap `max_model_len`, or add GPU capacity.
- **API gateway**: misconfigured rate limits generate a surge of 429 responses to valid users. The model server may appear healthy (GPU utilization normal, no errors), but end-users see errors. Fix: audit rate-limit policies per tenant, test with load generation tools before deploying a new policy.
- **Infrastructure**: a node being drained for maintenance or a pod being evicted due to resource pressure generates sudden 503 errors with no model-side log entries. Fix: set Pod Disruption Budgets (`maxUnavailable: 0` during low-traffic periods), use pod anti-affinity rules to spread replicas across nodes.
- **Observability**: Prometheus scrapes fail silently when a ServiceMonitor selector does not match the pod labels. You see no metrics, no alerts fire, and when a real incident happens you are blind. Fix: test scrape connectivity in staging, add a meta-alert that fires if the `up` metric for vLLM goes to 0.



## Key takeaways

1. The six layers of the serving stack are model artifact, inference runtime, serving framework, API gateway, infrastructure, and observability. A bottleneck at any single layer propagates to every request.

2. GPU memory is consumed by three things at serving time: model weights, KV cache, and activation buffers. KV cache grows linearly with batch size and sequence length — budget it before you deploy.

3. vLLM and Triton both cover layers 2 and 3, but they are not substitutes: vLLM is LLM-specific (PagedAttention, continuous batching, prefix caching); Triton is general-purpose (multi-framework, ensemble, TRT acceleration).

4. The API gateway (layer 4) should be kept separate from the serving framework (layer 3). Auth, rate limiting, and routing logic should not live inside vLLM or TorchServe.

5. Observability is not optional. Without layer-tagged metrics (TTFT, KV cache %, queue depth, auth latency), you will spend hours debugging with SSH and `nvidia-smi` instead of minutes with a Prometheus dashboard.

6. Disaggregated serving (split prefill/decode workers) doubles peak throughput for long-context LLM workloads at the cost of a KV transfer hop. Do not add this complexity for < 100 QPS.

7. Container images must pin the CUDA toolkit version. A mismatch between the image's CUDA and the host node's driver causes `libcuda.so` errors that look like hardware failures.

8. `initialDelaySeconds` on Kubernetes liveness probes must cover model load time (60–120 seconds for large models). Without this, Kubernetes kills the pod before it finishes loading.

9. The GPU memory budget equation — $M = M_{\text{weights}} + M_{\text{kvcache}} + M_{\text{activations}} + M_{\text{overhead}}$ — should be computed before every new deployment. For Llama-3-8B in BF16 at 256 concurrent sequences with 2048-token context: 16 GB weights + 32 GB KV cache + 2 GB overhead = 50 GB, which fits on H100 80 GB but not A100 40 GB.

10. Framework choice is a branching decision: LLM → vLLM or TGI; multi-model pipeline → Triton; general async Python → Ray Serve; quick path to production → BentoML.



## Further reading

- **vLLM paper**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. The foundational paper on PagedAttention and continuous batching.
- **Triton Inference Server documentation**: [docs.nvidia.com/deeplearning/triton-inference-server](https://docs.nvidia.com/deeplearning/triton-inference-server). Comprehensive reference for model repository layout, backends, and dynamic batching configuration.
- **Ray Serve documentation**: [docs.ray.io/en/latest/serve](https://docs.ray.io/en/latest/serve). Official guide to deployments, autoscaling, and DeploymentHandle composition.
- **FlashAttention-2**: Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," ICLR 2024. Explains the IO-aware attention algorithm that underpins vLLM and TGI throughput.
- **Within this series**: [what-is-model-serving](/blog/machine-learning/model-serving/what-is-model-serving) introduces the SLO triangle that this post builds on; [batching-fundamentals-latency-throughput-tradeoff](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) goes deep on the batching algorithms at layer 3; [vllm-deep-dive](/blog/machine-learning/model-serving/vllm-deep-dive) dissects the PagedAttention scheduler in detail; [the-model-serving-playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) provides the full decision tree from model to production.
