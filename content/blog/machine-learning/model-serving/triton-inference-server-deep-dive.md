---
title: "Triton Inference Server deep dive: multi-model serving at production scale"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master NVIDIA Triton Inference Server — from model repository layout and config.pbtxt anatomy to dynamic batching math, ensemble pipelines, concurrent GPU instances, perf_analyzer sweeps, and Prometheus dashboards."
tags:
  [
    "model-serving",
    "inference",
    "triton-inference-server",
    "tensorrt",
    "onnx-runtime",
    "dynamic-batching",
    "model-ensemble",
    "nvidia-gpu",
    "ml-infrastructure",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/triton-inference-server-deep-dive-1.png"
---

It is 2:47 AM on a Thursday when the p99 latency alert fires. Your ResNet-based fraud-detection service, which had been cruising at 800 req/s over the past three months, suddenly climbed to 1,600 req/s after the North-America settlement batch closed — and the TorchServe instance you jury-rigged six months ago is now serving requests at 340ms p99. The SLA is 50ms. Three of your six Python worker processes are stuck waiting on a lock inside the HTTP handler. The rest are fighting over a single GPU context because nobody configured concurrent execution. You check the NVIDIA System Management Interface — GPU utilization is 38%. The GPU is not busy; your serving infrastructure is the bottleneck. By 4:00 AM you have the alerts silenced, but the architecture is clearly broken.

This scenario is not exotic. It plays out in production dozens of times a year across teams that outgrow single-model serving frameworks. It is the exact failure mode that NVIDIA Triton Inference Server was built to eliminate. Triton is not just another model-serving framework — it is a purpose-built multi-model, multi-framework GPU inference system that treats batching, concurrency, ensembles, and observability as first-class concerns rather than bolt-ons. After years of production hardening at hyperscalers and enterprise ML teams, Triton is now the gold standard for high-throughput CV and NLP inference on NVIDIA GPUs.

By the end of this post you will know: exactly how Triton's model repository and `config.pbtxt` contract works, how dynamic batching queues requests and forms optimal batches, how to write the Python backend for custom preprocessing, how to build a TensorRT engine from an ONNX checkpoint, how to chain models into a DAG ensemble without a round-trip to the client, how to measure your system with `perf_analyzer`, how to deploy on Kubernetes, and when Triton is the right tool — and when it is not. The running example throughout is a ResNet-50 TensorRT engine and a BERT-base ONNX model deployed together on a single A100 40GB.

![Triton Inference Server layered architecture: HTTP/gRPC clients to GPU pool](/imgs/blogs/triton-inference-server-deep-dive-1.png)

Triton maps cleanly onto the series spine — Model → Packaging → Runtime → Server → Infrastructure → Observability. It sits at the Runtime and Server layers, consuming packaged model artifacts (TRT engine plans, ONNX files, TorchScript archives) and exposing a uniform gRPC/HTTP API above them. Every design decision in Triton is a trade on the SLO triangle: latency ↔ throughput ↔ cost. You will see exactly how each knob moves you around that triangle.

A note on what makes Triton architecturally distinct from its peers: TorchServe is a model registry and HTTP wrapper — it does not schedule batches across multiple model instances or compose models into pipelines. Ray Serve is an async actor system — powerful for Python-native serving and autoscaling, but not optimized for the GPU kernel dispatch layer. BentoML is a packaging and deployment wrapper — it can target Triton as a runtime but does not expose Triton's internal knobs. Triton is the only framework that works at the GPU scheduling level: it understands CUDA streams, instance groups, TRT backend state, and pinned memory pools as first-class scheduling resources. This is why, for multi-model GPU serving, Triton has no real competitors — and why it requires the configuration discipline to operate correctly.

## 1. The model repository: Triton's filesystem contract

Triton does not have a CLI import step or a registry API. It discovers models by scanning a directory tree with a fixed layout. This simplicity is a feature: you deploy a new model by copying files, and you roll it back by deleting the version directory — no state machine to debug, no registration call to undo.

The canonical layout looks like this:

```
model_repository/
├── resnet50_trt/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan          # TensorRT serialized engine
├── bert_base_onnx/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── preprocess_python/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── classify_ensemble/
    ├── config.pbtxt
    └── 1/
        └── (no artifact required — ensemble is pure routing config)
```

The rules are strict and non-negotiable:

1. The top-level directory name becomes the model's name in the Triton API. This name is used in every client call, in `perf_analyzer`, and in Prometheus metrics labels. Choose it carefully.
2. Subdirectories are integer version numbers (`1`, `2`, `3`). Triton loads the highest version by default, or follows the version policy set in `config.pbtxt`. Gaps are allowed — version 1 and 3 without version 2 is valid.
3. Each version contains exactly one artifact: `model.plan` (TensorRT), `model.onnx` (ONNX Runtime), `model.pt` (TorchScript/LibTorch), `model.py` (Python backend), `model.savedmodel/` (TensorFlow SavedModel directory). The filename is fixed — Triton does not accept arbitrary filenames.
4. `config.pbtxt` sits next to the version directories, not inside them. It is shared across all versions unless you need per-version configuration (which requires separate model directories).

![Triton model repository directory tree with versioned artifacts](/imgs/blogs/triton-inference-server-deep-dive-2.png)

**Version policies**: Triton's default is `POLICY_LATEST` — it loads and serves only the highest-numbered version. Three other policies are available: `POLICY_ALL` (serve every version simultaneously, selectable by the client), `POLICY_SPECIFIC` (serve a named list of versions), and `POLICY_NONE` (no version-based filtering). For production A/B testing, `POLICY_ALL` combined with two named model variants (e.g. `resnet50_trt_v1`, `resnet50_trt_v2`) and an upstream load balancer is cleaner than `POLICY_SPECIFIC` if you want traffic-splitting logic outside Triton.

**Model control mode**: by default Triton loads all models it finds at startup and hot-reloads when it detects a change in the repository (via polling or inotify). You can switch to explicit control mode with `--model-control-mode=explicit`, which requires a management API call to load each model. This is safer in production — you decide when to activate a new model version rather than having Triton auto-load on filesystem change.

**Remote model repositories**: Triton supports AWS S3 (`s3://bucket/prefix`), Google Cloud Storage (`gs://bucket/prefix`), and Azure Blob Storage as model repository backends. The server downloads artifacts on demand. For large TRT engines (>500MB), set `--model-load-retry-count` and ensure your container has sufficient ephemeral storage.

## 2. The config.pbtxt: anatomy of the inference contract

The `config.pbtxt` is a protobuf text format file. Every important field deserves careful attention because errors here will either silently degrade performance or cause Triton to refuse to load your model. Getting it right is the difference between 38% and 94% GPU utilization.

Here is a complete, annotated `config.pbtxt` for the ResNet-50 TensorRT engine:

```protobuf
name: "resnet50_trt"
platform: "tensorrt_plan"
max_batch_size: 64

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 32, 64 ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    kind: KIND_GPU
    count: 4
    gpus: [ 0 ]
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator: [
      {
        name: "tensorrt"
        parameters {
          key: "precision_mode"
          value: "FP16"
        }
      }
    ]
  }
}
```

Walk through each field in depth:

**`platform`** specifies which backend handles this model. The canonical values are `tensorrt_plan`, `onnxruntime_onnx`, `pytorch_libtorch`, `python`, `tensorflow_savedmodel`. In Triton 23.x+, the preferred approach is `backend: "tensorrt"` instead of `platform`, but both work and `platform` remains supported. The backend value also accepts custom backend names for third-party backends registered via the backend API.

**`max_batch_size`** is not the batch size Triton always uses — it is the absolute maximum the backend will accept in a single kernel launch. For TensorRT engines built with a fixed max batch size, this value must match the engine's max batch exactly; mismatches cause a load failure. For ONNX models with dynamic batch axes, this caps the scheduler's maximum dispatch size. Setting it too high with TensorRT can OOM your GPU on the first large-batch request; setting it too low leaves throughput on the table. A good starting point is 64 for CV models on an A100 — tune with `perf_analyzer` afterwards.

**`input` and `output`** define the tensor contract between the client and the backend. The `dims` field does NOT include the batch dimension — Triton prepends it automatically. So `dims: [ 3, 224, 224 ]` means each individual input image is 3×224×224; a batched input tensor will be `[batch_size, 3, 224, 224]`. Use `-1` for dynamic dimensions where the size is unknown at config time (variable-length sequences, for instance). `data_type` follows Triton's type vocabulary: `TYPE_FP32`, `TYPE_FP16`, `TYPE_INT64`, `TYPE_INT32`, `TYPE_UINT8`, `TYPE_BYTES` (for string/binary tensors), `TYPE_BOOL`. Mismatched data types between the config and what the client sends cause silent cast overhead or outright errors.

**`dynamic_batching`** is where throughput lives. `preferred_batch_size: [32, 64]` tells the scheduler to form batches of 32 or 64 if possible. Providing a list lets the scheduler target the smaller size first (filling a batch of 32 faster at low QPS) and ramp to 64 under high load. `max_queue_delay_microseconds: 5000` means Triton will wait up to 5ms for more requests before dispatching a partial batch. This is the primary latency-throughput knob and deserves a full section of its own.

**`instance_group`** controls concurrency. `kind: KIND_GPU` means GPU execution; `KIND_CPU` allocates CPU threads for model execution; `KIND_AUTO` lets Triton decide. `count: 4` spawns four independent model instances — each gets its own GPU memory allocation and its own CUDA stream, enabling true parallel execution. `gpus: [0]` pins all instances to GPU 0; `gpus: [0, 1]` would distribute two instances per GPU if count were 4. To pin instance 1 to GPU 0 and instance 2 to GPU 1, use two separate `instance_group` blocks.

**`optimization.execution_accelerators`** is where you enable TensorRT layer fusion, FP16, and INT8 precision on top of the ONNX Runtime backend. For ONNX models you can enable the TensorRT execution provider here without converting to a `.plan` file first — ORT will JIT-compile the subgraphs using TRT at first inference. For native TRT backends, this block can tune workspace size and calibration cache paths.

**`version_policy`**: add `version_policy { latest { num_versions: 2 } }` to keep the two most recent versions live simultaneously — useful during staged rollouts where you want instant rollback capability.

For BERT-base ONNX the config carries multiple inputs (tokenizer outputs) and a longer batching delay:

```protobuf
name: "bert_base_onnx"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 128 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 128 ]
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    dims: [ 128 ]
  }
]

output [
  {
    name: "last_hidden_state"
    data_type: TYPE_FP32
    dims: [ 128, 768 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 8000
}

instance_group [
  {
    kind: KIND_GPU
    count: 2
    gpus: [ 0 ]
  }
]
```

BERT uses longer batching delays (`8ms`) because sequence inference is more compute-intensive and the throughput gain from full batches is larger. Two instances instead of four because each BERT instance at batch 32 requires roughly 4.8GB of GPU memory — four instances would consume ~19GB for BERT alone, cutting into the ResNet budget and the I/O pool.

## 3. Building TensorRT engines from ONNX

Before Triton can serve a TRT model, you need to compile the ONNX export into a `.plan` file. The `trtexec` tool ships with every TensorRT installation and handles the conversion for most standard architectures.

```bash
# Step 1: Export PyTorch model to ONNX
python3 - <<'EOF'
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True).eval().cuda()
dummy = torch.randn(1, 3, 224, 224).cuda()

torch.onnx.export(
    model,
    dummy,
    "resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17
)
print("ONNX export complete")
EOF

# Step 2: Compile ONNX to TRT engine (FP16, max batch 64)
trtexec \
  --onnx=resnet50.onnx \
  --saveEngine=model_repository/resnet50_trt/1/model.plan \
  --fp16 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:32x3x224x224 \
  --maxShapes=input:64x3x224x224 \
  --workspace=4096 \
  --verbose

# For INT8 with calibration:
trtexec \
  --onnx=resnet50.onnx \
  --saveEngine=model_repository/resnet50_trt/1/model.plan \
  --int8 \
  --fp16 \
  --calib=calib_cache.bin \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:32x3x224x224 \
  --maxShapes=input:64x3x224x224 \
  --workspace=8192
```

The `--minShapes`, `--optShapes`, `--maxShapes` triplet is critical for dynamic-shape TRT engines. TRT builds a separate optimization profile for each shape range. The `--optShapes` value is the "most common" shape — TRT optimizes most aggressively for this. Build time is proportional to the number of layers and the shape range width; a wider shape range means TRT must keep more kernel variants resident. For fixed-production workloads where batch size is known, omitting the `min`/`opt`/`max` triplet and using `--explicitBatch --shapes=input:32x3x224x224` locks the engine to exactly one shape and delivers peak performance.

**The `.plan` file is GPU-architecture-specific.** A plan compiled on an A100 (sm_80) will fail to load on a T4 (sm_75). Your CI pipeline must build separate plans for each GPU SKU you deploy to. A common pattern is to store plans under architecture-namespaced S3 paths: `s3://ml-artifacts/resnet50_trt/sm_80/v1/model.plan`, and have deployment scripts pull the correct plan at startup.

## 4. Backends: choosing the right execution engine

Triton supports six first-party backends and allows custom backends via a C API. Understanding the trade-offs is critical for matching your model and workload to the right executor.

**TensorRT (`tensorrt_plan`)** is NVIDIA's layer-fusion compiler. It takes an ONNX model (or a TRT network definition API) and produces a `.plan` file — a serialized, device-specific optimized engine. The plan file is NOT portable across GPU architectures (a T4 plan will crash on an A100). TensorRT delivers the highest throughput of any backend, typically 3–5× faster than ONNX Runtime for CV workloads, through kernel fusion, INT8/FP16 calibration, and Winograd convolution. The cost is build time (10–30 minutes for large models), a rebuild whenever the GPU architecture changes, and reduced flexibility for dynamic input shapes.

**ONNX Runtime (`onnxruntime_onnx`)** is the most portable backend. A model exported via `torch.onnx.export()` can run unchanged on T4, A100, H100, or CPU. ORT with the CUDA execution provider achieves 60–80% of TensorRT throughput on typical ResNet workloads with zero rebuild cost. ORT also supports the TensorRT EP, which JIT-compiles TRT subgraphs on first execution — a pragmatic middle ground between full TRT builds and plain CUDA.

**PyTorch LibTorch (`pytorch_libtorch`)** runs TorchScript `.pt` files using the C++ LibTorch runtime. Useful when your model uses dynamic control flow that ONNX cannot represent — branching `if` statements, variable-length loops, Python dataclasses in the forward pass. The performance is similar to eager PyTorch — LibTorch does not apply kernel fusion. Use this when you cannot cleanly TorchScript your model or when you need `torch.compile` output (though `torch.compile` with `mode="reduce-overhead"` produces eager-compatible bytecode, not standard TorchScript `.pt` format).

**Python backend (`python`)** is the escape hatch. You write a `model.py` with three lifecycle methods: `initialize()`, `execute(requests)`, and `finalize()`. Triton calls these methods and handles tensor I/O marshalling. Execution overhead comes from crossing the Python-C++ boundary on every request and from Python's GIL serializing concurrent calls within one model instance. Throughput is typically 5–10× lower than TensorRT for pure compute, but for preprocessing pipelines, custom tokenizers, or models that require Python-only libraries, it is the only viable option without custom C++ backend work.

**vLLM backend**: since Triton 24.04, there is an experimental vLLM backend that runs vLLM's `AsyncLLMEngine` inside Triton, surfacing PagedAttention and continuous batching through the standard Triton gRPC/HTTP API. This is covered in the dedicated [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) post. For most LLM serving you should use vLLM directly — the Triton vLLM backend adds API compatibility overhead and is not recommended for production LLM serving above ~50 concurrent requests.

![Backend selection matrix: TensorRT vs ONNX vs LibTorch vs Python](/imgs/blogs/triton-inference-server-deep-dive-6.png)

**Selection heuristic**:
- New production CV/audio workload on NVIDIA GPU: start with ONNX Runtime ORT for dev velocity, switch to TRT when you need the last 30–40% throughput or when latency SLAs tighten.
- Model has dynamic shapes, custom Python ops, or dynamic control flow: Python backend or LibTorch. Accept the throughput penalty; profile to confirm it is acceptable.
- LLM inference: use vLLM directly, not Triton (unless you need the Triton ensemble API for a multi-model pipeline around an LLM).
- CPU-only deployment: ONNX Runtime with the OpenVINO or DNNL execution provider (Triton's CPU backend wraps ORT; you can also use OpenVINO model server for pure CPU scenarios with better support).

## 5. The Python backend: writing model.py

The Python backend deserves its own treatment because it is the entry point for any custom preprocessing or postprocessing logic that cannot be expressed as a standard DNN. Here is a complete, production-grade `model.py` for a JPEG-to-tensor preprocessing step:

```python
# preprocess_python/1/model.py
import triton_python_backend_utils as pb_utils
import numpy as np
import io
from PIL import Image
import torchvision.transforms as T


class TritonPythonModel:
    """Preprocessing: JPEG bytes → normalized float32 tensor [1, 3, 224, 224]."""

    def initialize(self, args):
        """Called once at model load time. Set up transforms and constants."""
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        # Log the model configuration for debugging
        import json
        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(
            model_config, "preprocessed"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )

    def execute(self, requests):
        """
        Called for each batch of requests. Must return a response for each request.
        requests: list of pb_utils.InferenceRequest
        """
        responses = []

        for request in requests:
            # Extract the raw image bytes tensor
            raw_input = pb_utils.get_input_tensor_by_name(request, "raw_input")
            image_bytes = raw_input.as_numpy()   # shape: [1, N] where N = byte length

            # Decode JPEG bytes
            img_bytes = bytes(image_bytes[0].tobytes())
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Apply transforms: PIL Image → float32 tensor [3, 224, 224]
            tensor = self.transform(img).numpy()            # [3, 224, 224]
            tensor = tensor[np.newaxis, ...]                # [1, 3, 224, 224]
            tensor = tensor.astype(self.output_dtype)

            # Build the output tensor
            output_tensor = pb_utils.Tensor("preprocessed", tensor)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        """Called at model unload time. Release resources."""
        pass
```

Key points about the Python backend API:

- `pb_utils.get_input_tensor_by_name(request, name)` fetches a named input tensor from one request object. The tensor name must match `config.pbtxt`.
- `.as_numpy()` converts the Triton tensor to a numpy array. For string/bytes tensors, each element is a Python `bytes` object.
- `pb_utils.Tensor(name, numpy_array)` wraps a numpy array as a Triton output tensor.
- You must return exactly one `InferenceResponse` per `InferenceRequest`. If processing fails, use `pb_utils.InferenceResponse(error=pb_utils.TritonError("message"))`.
- The `execute()` function receives a list (not a true batch ndarray) — the batching is logical. You loop over each request individually and build individual responses. If you want to batch-process them (e.g., running a vectorized numpy transform over all images at once), you must manually stack the inputs.

The Python backend config for this preprocessor:

```protobuf
name: "preprocess_python"
backend: "python"
max_batch_size: 32

input [
  {
    name: "raw_input"
    data_type: TYPE_BYTES
    dims: [ 1 ]
  }
]

output [
  {
    name: "preprocessed"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

instance_group [
  {
    kind: KIND_CPU
    count: 4
  }
]
```

Note `KIND_CPU` — preprocessing (JPEG decode, PIL resize) is CPU-bound, not GPU-bound. Running four CPU instances allows parallel preprocessing while the GPU is busy with the inference step. This is the Triton pattern for effective pipeline parallelism: CPU preprocessing instances feeding GPU inference instances.

## 6. Dynamic batching: the queuing theory

Dynamic batching is Triton's most important performance feature. Without it, each client request would launch a separate GPU kernel — and GPU kernel launch overhead (~5–20µs) combined with memory bandwidth saturation from tiny single-image batches destroys throughput.

The mechanics work like this: Triton maintains a per-model **scheduler queue**. Incoming requests are enqueued immediately. The scheduler evaluates two conditions on each cycle:

1. Has the queue accumulated enough requests to fill `preferred_batch_size`?
2. Has the oldest request in the queue been waiting longer than `max_queue_delay_microseconds`?

If either condition is true, the scheduler dequeues up to `max_batch_size` requests, concatenates their input tensors along the batch dimension, and dispatches one kernel launch to the backend.

![Dynamic batching queue: preferred_batch_size vs timeout](/imgs/blogs/triton-inference-server-deep-dive-3.png)

The performance math here is worth doing explicitly. For a ResNet-50 TRT engine on A100:

- Single-image inference: ~0.4ms kernel time + ~0.018ms launch overhead = **0.418ms**
- Batch-64 inference: ~1.4ms kernel time + ~0.018ms launch overhead = **1.418ms**
- Throughput at batch 1: $1000 / 0.418 = 2{,}392$ images/s
- Throughput at batch 64: $64 \times 1000 / 1.418 = 45{,}134$ images/s — an **18.9× improvement**

This is the arithmetic behind dynamic batching. The GPU does not care about your request rate; it cares about the compute-to-launch-overhead ratio, which is proportional to batch size. The `max_queue_delay_microseconds` is your SLA tax: every microsecond of delay you are willing to accept converts into more batching opportunity.

Setting the delay requires knowing your arrival rate. If requests arrive at $\lambda$ req/s and you want to fill a batch of $B$, the expected wait time for Poisson arrivals is approximately $B / \lambda$ seconds. So:

$$\text{max\_queue\_delay} \approx \frac{B}{\lambda}$$

For $\lambda = 500$ req/s and $B = 32$: $32 / 500 = 64$ms — too high for interactive use. You would reduce $B$ to 8 or set the delay to 10ms and accept partial batches of roughly $0.010 \times 500 = 5$ requests on average. This is the exact trade on the latency ↔ throughput side of the SLO triangle.

**Little's Law applied to Triton's queue**: Little's Law states that in a stable queuing system, $L = \lambda W$, where $L$ is the average number of requests in the system, $\lambda$ is the arrival rate, and $W$ is the average time a request spends in the system (queue + service). For Triton's dynamic batcher:

$$W_{\text{queue}} = \frac{L_{\text{queue}}}{\lambda}$$

If the queue holds an average of 10 requests and arrivals are 1,000 req/s, the average queue wait is 10ms. This means queue depth monitoring (via `nv_inference_pending_request_count` in Prometheus) is a leading indicator of latency trouble — a growing queue warns you that the backend cannot keep up before p99 latency climbs past the SLA.

**Sequence batching** is a distinct scheduler for stateful models — RNNs, streaming ASR, session-aware ranking models. It routes requests from the same session (identified by a `correlation_id` field in the inference request header) to the same model instance, maintaining hidden state across calls. Configure it with:

```protobuf
sequence_batching {
  max_sequence_idle_microseconds: 5000000    # 5s idle → release instance
  control_input [
    {
      name: "START"
      control [{ kind: CONTROL_SEQUENCE_START, fp32_false_true: [0, 1] }]
    },
    {
      name: "END"
      control [{ kind: CONTROL_SEQUENCE_END, fp32_false_true: [0, 1] }]
    }
  ]
}
```

Sequence batching is mutually exclusive with dynamic batching — you pick one scheduler per model.

**Batching under bursty traffic**: real production traffic is not Poisson — it arrives in bursts driven by user behavior (ad campaigns firing, API integrations triggering simultaneously). Under bursty arrivals, the queue can grow faster than the backend drains it. The correct way to handle this is not to increase `max_queue_delay_microseconds` (which would hurt your p50 latency during normal traffic) but to pre-size your instance count for peak load and rely on the KEDA autoscaler to bring up additional pods during sustained bursts. The queue delay knob is for steady-state batching efficiency; instance count and pod scaling are for burst capacity.

A useful heuristic for burst sizing: examine your traffic histograms for the 99th percentile 10-second burst rate (not just the average). If your p99-10s burst is 3× your average QPS, size your instance count for 3× your normal working concurrency, with the autoscaler covering any further spikes.

**Optimal batch size derivation for a real model**: for a transformer model where attention is $O(n^2)$ in sequence length, the marginal compute cost of each additional sequence in a batch grows quadratically — which means the throughput gain from batching flattens faster than for a pure FLOP-bound convolution model. The inflection point occurs at the batch size where the scheduler's batching latency overhead (the queue delay plus tensor concatenation time) equals the marginal compute gain:

$$\frac{\partial T_{\text{compute}}(B)}{\partial B} \approx \frac{T_{\text{overhead}}}{T_{\text{compute}}(1)}$$

For BERT-base at sequence 128, empirically this inflection is around batch 32–64 (beyond batch 64, throughput gains flatten to under 5% per doubling). For ResNet-50 (FLOP-bound, not memory-bound), the inflection is around batch 64–128.

## 7. Launching Triton in Docker

All production Triton deployments use NVIDIA's official NGC containers. The container tag corresponds to Triton's release version and pins the included CUDA, TRT, ONNX, and Python versions — critical for reproducibility.

All production Triton deployments use NVIDIA's official NGC containers. The container tag corresponds to Triton's release version and pins the included CUDA, TRT, ONNX, and Python versions — critical for reproducibility.

```bash
# Pull the latest Triton NGC container (matching your CUDA driver version)
docker pull nvcr.io/nvidia/tritonserver:24.05-py3

# Launch Triton pointing at your model repository
docker run --gpus=all \
  --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /mnt/model-repo:/models \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver \
    --model-repository=/models \
    --strict-model-config=false \
    --log-verbose=1 \
    --cuda-memory-pool-byte-size=0:12000000000 \
    --pinned-memory-pool-byte-size=4000000000 \
    --model-control-mode=explicit
```

Key flags explained:

- `--model-repository`: path inside the container to the model directory. Use a volume mount from your NFS or object-store mount point. Multiple repositories are supported: `--model-repository=/models/cv --model-repository=/models/nlp`.
- `--strict-model-config=false`: Triton will auto-infer tensor shapes from the model artifact when `config.pbtxt` omits them. Useful during development; disable in production (explicit configs catch bugs at load time, not at inference time when the first malformed request arrives).
- `--cuda-memory-pool-byte-size=0:12000000000`: pre-allocates 12GB of CUDA device memory on GPU 0 for inference buffers. Format is `gpu_id:bytes`. This prevents fragmentation during rapid request bursts when thousands of requests arrive within one second.
- `--pinned-memory-pool-byte-size=4000000000`: pre-allocates 4GB of CPU pinned memory for zero-copy host-to-device DMA transfers. Critical for high-throughput image ingestion where you cannot afford the `cudaMallocHost` overhead on the hot path.
- `--shm-size=16g`: shared memory needed when using CUDA IPC between processes or when the Python backend spawns subprocess workers.
- `--model-control-mode=explicit`: models must be explicitly loaded via the management API. This is the safer production mode.

**Loading models via the management API** after startup in explicit mode:

```bash
# Load a model
curl -X POST "localhost:8000/v2/repository/models/resnet50_trt/load"

# Unload a model (does not delete files)
curl -X POST "localhost:8000/v2/repository/models/resnet50_trt/unload"

# Check server status
curl "localhost:8000/v2/health/ready"

# List loaded models
curl "localhost:8000/v2/repository/index"
```

The management API is gRPC-native too, but the HTTP endpoints are convenient for deployment scripts and Kubernetes init containers.

## 8. Model ensemble: chaining models without round-trips

An ensemble model in Triton is a DAG of model calls defined entirely in `config.pbtxt`. No artifact binary is needed — the ensemble config is pure routing logic. When a client sends one request to `classify_ensemble`, Triton orchestrates all the DAG hops internally, passing GPU tensors directly between steps without copying to CPU or traversing a network socket.

![ResNet-50 + BERT ensemble pipeline in Triton](/imgs/blogs/triton-inference-server-deep-dive-4.png)

The performance benefit is significant. Without ensembles, a client would need to: send an image, receive ResNet features, send features to BERT, receive logits, send logits to postprocessing. Each hop is a gRPC round-trip — at 0.5ms RTT in a co-located datacenter (and 10–50ms cross-datacenter), three hops add 1.5ms to 150ms of pure latency tax with zero compute. With ensemble, all four hops happen inside Triton with direct CUDA memory transfers via pinned buffer pointers.

The ensemble `config.pbtxt` for a preprocessing → ResNet → BERT → postprocessing pipeline:

```protobuf
name: "classify_ensemble"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "raw_image"
    data_type: TYPE_BYTES
    dims: [ 1 ]
  }
]

output [
  {
    name: "top5_labels"
    data_type: TYPE_BYTES
    dims: [ 5 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocess_python"
      model_version: -1
      input_map {
        key: "raw_input"
        value: "raw_image"
      }
      output_map {
        key: "preprocessed"
        value: "img_tensor"
      }
    },
    {
      model_name: "resnet50_trt"
      model_version: -1
      input_map {
        key: "input"
        value: "img_tensor"
      }
      output_map {
        key: "output"
        value: "resnet_features"
      }
    },
    {
      model_name: "bert_base_onnx"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "resnet_features"
      }
      output_map {
        key: "last_hidden_state"
        value: "bert_output"
      }
    },
    {
      model_name: "postprocess_python"
      model_version: -1
      input_map {
        key: "logits"
        value: "bert_output"
      }
      output_map {
        key: "top5_labels"
        value: "top5_labels"
      }
    }
  ]
}
```

Key points about ensemble configs:

- `input_map` and `output_map` wire outputs to inputs using string names. The `key` is the tensor name as defined in the sub-model's `config.pbtxt`; the `value` is an internal ensemble wire name you invent. The wire name must be unique across all steps — reusing the same wire name for different outputs causes the second assignment to silently overwrite the first.
- DAGs (not just linear chains) are fully supported. You can fan out from one model to two parallel models and fan in their outputs to a third. Parallel branches execute concurrently when they have no data dependency.
- Each step runs the sub-model's full dynamic batching policy. The ensemble scheduler applies batching at the ensemble boundary, and each step's sub-model may independently batch its dispatch.
- `model_version: -1` means "latest version." For staged rollouts, pin a specific version number so a background version bump doesn't change ensemble behavior.

**Ensemble limitations**: Triton ensembles cannot loop — the DAG must be acyclic. Conditional steps (execute step B only if step A output meets a threshold) are not supported natively in the ensemble config. For conditional logic, use the Python backend as an ensemble step that implements the condition.

#### Worked example: image classification pipeline on A100

**Setup**: ResNet-50 TensorRT FP16 + BERT-base ONNX FP32, single A100 40GB, ensemble batch size 16, client and server co-located in the same datacenter rack.

**Without ensemble** (client orchestrated, three gRPC round-trips):
- Preprocessing gRPC request + response: 0.8ms (includes Python backend execution)
- Network RTT to ResNet: 0.5ms overhead
- ResNet inference: 1.4ms
- Network RTT to BERT: 0.5ms overhead
- BERT inference: 3.8ms
- Network RTT for result: 0.3ms
- **Total e2e**: 7.3ms per request at batch 16

**With ensemble** (Triton internal DAG, zero network hops):
- Preprocessing Python backend: 0.9ms
- In-memory tensor handoff (pinned CUDA buffer): ~0.05ms
- ResNet TRT dispatch: 1.4ms
- In-memory tensor handoff: ~0.05ms
- BERT ONNX dispatch: 3.8ms
- Postprocessing Python: 0.4ms
- **Total e2e**: 6.6ms per request at batch 16

The latency saving is 700µs on a local rack — meaningful but not dramatic. The real win emerges in two scenarios: cross-datacenter pipelines (saves 30–150ms per hop), and rate-limiting scenarios where the client would otherwise need multiple API keys and multiple connections to chain models independently.

## 9. Concurrent model execution: filling the GPU

A single model instance, even with dynamic batching, cannot saturate a modern A100's 312 TFLOPS of FP16 compute. The reason is the kernel execution gap: while Triton is receiving the next batch of requests, serializing tensors, and calling the CUDA runtime's kernel launch infrastructure, the GPU SMs are idle. Multiple concurrent instances overlap these gaps by keeping the GPU dispatch pipeline continuously fed.

The `instance_group` config is how you set concurrency:

```protobuf
instance_group [
  {
    kind: KIND_GPU
    count: 4
    gpus: [ 0 ]
  }
]
```

Each instance gets its own GPU memory allocation and its own CUDA stream. CUDA streams allow multiple kernel launches to overlap execution when the kernels are small enough to share SMs — a key property of ResNet's many small convolutional kernels. For large kernel models (e.g., matrix multiplications in transformer attention layers), concurrent instances have less SM overlap benefit but still eliminate the inter-batch idle time.

![Concurrent instances: 1 instance vs 4 instances on A100](/imgs/blogs/triton-inference-server-deep-dive-5.png)

**GPU MIG partitioning**: NVIDIA's Multi-Instance GPU feature divides one physical A100 into up to seven isolated MIG instances (on the A100 80GB: 1×7g.40gb, 2×3g.20gb, 3×2g.20gb, 7×1g.5gb, etc.), each with dedicated SM count, L2 cache partition, and DRAM bandwidth slice. Triton supports MIG via the GPU UUID syntax:

```protobuf
instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: ["MIG-GPU-abc12345-0000-0000-0000-000000000000/0/0"]
  }
]
```

MIG is useful when you have SLA isolation requirements — a critical real-time inference endpoint should not be able to steal GPU SM time from a background batch pipeline. The trade-off: each MIG instance has lower peak throughput than the full GPU, and you cannot use MIG and time-slicing simultaneously on the same physical GPU.

**GPU memory math for sizing**:

$$\text{VRAM} = N_{\text{instances}} \times (W_{\text{model}} + B_{\text{activations}} + B_{\text{IO buffers}})$$

Where:
- $W_{\text{model}}$: model weights in bytes (FP16: param count × 2; INT8: param count × 1)
- $B_{\text{activations}}$: peak activation memory during a forward pass at max batch size
- $B_{\text{IO buffers}}$: input + output tensor buffers, sourced from the CUDA pool

For ResNet-50 FP16 at batch 64: $98\text{MB} + 50\text{MB} + 20\text{MB} = 168\text{MB}$ per instance. Four instances = 672MB — trivial on a 40GB A100. For BERT-base ONNX FP32 at batch 32: $440\text{MB} + 4.2\text{GB} + 160\text{MB} = 4.8\text{GB}$ per instance. Two instances = 9.6GB.

For the A100 40GB with ResNet (×4) + BERT (×2):

| Component | VRAM usage |
|---|---|
| ResNet-50 TRT FP16 weights × 4 | 392 MB |
| ResNet activation buffers × 4 | 200 MB |
| BERT-base ONNX FP32 weights × 2 | 880 MB |
| BERT activation buffers (batch 32) × 2 | 8.4 GB |
| CUDA I/O pool | 2 GB |
| CUDA context overhead | ~1 GB |
| **Total** | **~13 GB** |

This leaves 27GB of headroom — enough to add additional ensemble variants for A/B testing or to load a DistilBERT alternative for lower-latency fallback.

## 10. Performance Analyzer: measuring your server

`perf_analyzer` is Triton's built-in load generator and measurement tool. It ships inside the `nvcr.io/nvidia/tritonserver` container and sends concurrent requests to your running Triton instance, measuring throughput and latency across a configurable concurrency range. No external load generator needed.

```bash
# Run inside the Triton container or install triton-client:
# pip install tritonclient[grpc] perf-analyzer

perf_analyzer \
  -m resnet50_trt \
  --concurrency-range 1:16:2 \
  --measurement-interval 10000 \
  --percentile 99 \
  --input-data /data/sample_inputs.json \
  -u localhost:8001 \
  -i grpc \
  --output-shared-memory none
```

The `--concurrency-range 1:16:2` sweeps from 1 to 16 concurrent clients with a step of 2. Each concurrency level runs for `--measurement-interval` milliseconds (10 seconds here) before advancing to the next level — long enough for the queue to stabilize. The `--percentile 99` flag makes the output show p99 latency alongside p50 and mean.

Sample output for ResNet-50 TRT on an A100 with 4 instances:

```
Concurrency: 1,  throughput: 183.4 infer/sec, latency p50: 5.2 ms, p99: 5.4 ms
Concurrency: 4,  throughput: 692.1 infer/sec, latency p50: 5.7 ms, p99: 5.9 ms
Concurrency: 8,  throughput: 1248.6 infer/sec, latency p50: 6.2 ms, p99: 7.2 ms
Concurrency: 12, throughput: 1582.3 infer/sec, latency p50: 7.8 ms, p99: 13.8 ms
Concurrency: 16, throughput: 1591.7 infer/sec, latency p50: 9.1 ms, p99: 27.6 ms
```

The inflection point between concurrency 8 and 12 is where GPU saturation occurs. Throughput plateaus at approximately 1,590 infer/s while p99 nearly doubles — this is the knee of the latency-throughput curve.

![perf_analyzer concurrency sweep: throughput plateau at concurrency-16](/imgs/blogs/triton-inference-server-deep-dive-7.png)

**Reading the sweep**: the optimal operating point is just before the knee — concurrency 8 in this case, delivering 1,248 infer/s at 7.2ms p99. Running at concurrency 16 for the marginal extra 343 infer/s costs 20ms of p99 latency headroom. If your SLA is 50ms you have plenty of room; if it is 10ms, concurrency 8 is your ceiling.

**Generating synthetic input data** for `perf_analyzer`:

```python
import json
import numpy as np
import base64

# Generate a JSON input file with 10 sample batches
samples = []
for _ in range(10):
    img = np.random.rand(3, 224, 224).astype(np.float32)
    samples.append({
        "data": [{"b64": base64.b64encode(img.tobytes()).decode()}]
    })

with open("/data/sample_inputs.json", "w") as f:
    json.dump({"data": [s["data"] for s in samples]}, f)
```

**Profiling the ensemble**: run `perf_analyzer -m classify_ensemble` to measure the full pipeline. Add `--trace-file=/tmp/trace.json` to capture a per-request trace that shows time spent in each ensemble step. This is the fastest way to identify which step is the bottleneck — preprocessing CPU time vs ResNet GPU time vs BERT GPU time.

**Calibrating `preferred_batch_size` from sweep data**: run three separate `perf_analyzer` runs varying `preferred_batch_size` in `config.pbtxt` (8, 16, 32, 64), restart Triton between runs, and compare throughput at the same concurrency level. A batch size of 32 that achieves 87% of batch-64 throughput with 40% lower p99 latency is typically the production winner — you capture most of the batching benefit at a lower latency cost.

## 11. Triton client libraries: calling the server from Python

The Python Triton client is the production interface for most application teams. Install via:

```bash
pip install tritonclient[grpc,http]
```

**gRPC client** (preferred for throughput — lower serialization overhead than HTTP/JSON):

```python
import tritonclient.grpc as grpcclient
import numpy as np

def run_resnet_inference(image_batch: np.ndarray) -> np.ndarray:
    """
    image_batch: float32 array, shape [N, 3, 224, 224]
    returns: float32 logits, shape [N, 1000]
    """
    client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

    inputs = [grpcclient.InferInput("input", image_batch.shape, "FP32")]
    inputs[0].set_data_from_numpy(image_batch)

    outputs = [grpcclient.InferRequestedOutput("output")]

    result = client.infer(
        model_name="resnet50_trt",
        inputs=inputs,
        outputs=outputs,
        model_version="1",
        timeout=5.0
    )

    return result.as_numpy("output")
```

**Async gRPC with callbacks** for high-concurrency pipelines:

```python
import tritonclient.grpc as grpcclient
import threading
import numpy as np

def run_async_batch(client, image_list):
    """Send N images concurrently via async callbacks."""
    results = {}
    lock = threading.Lock()
    event = threading.Event()
    inflight = [len(image_list)]

    def callback(result, error):
        with lock:
            if error:
                print(f"Error: {error}")
            else:
                req_id = result.get_response().id
                results[req_id] = result.as_numpy("output")
            inflight[0] -= 1
            if inflight[0] == 0:
                event.set()

    for i, img in enumerate(image_list):
        inp = [grpcclient.InferInput("input", [1, 3, 224, 224], "FP32")]
        inp[0].set_data_from_numpy(img[np.newaxis])
        out = [grpcclient.InferRequestedOutput("output")]
        client.async_infer(
            model_name="resnet50_trt",
            inputs=inp,
            outputs=out,
            callback=callback,
            request_id=f"req-{i}"
        )

    event.wait(timeout=60.0)
    return results

client = grpcclient.InferenceServerClient("localhost:8001")
logits = run_async_batch(client, image_list)
```

The async pattern multiplies throughput: a synchronous loop over 1,000 images at 5ms each takes 5 seconds. With 16 in-flight async requests, it completes in approximately $1000 / 16 \times 5\text{ms} = 312\text{ms}$ — a 16× wall-clock improvement.

**Health and readiness checks** for Kubernetes probes:

```python
client = grpcclient.InferenceServerClient("localhost:8001")

# Liveness probe (server process is running)
print(client.is_server_live())

# Readiness probe (all models loaded and ready)
print(client.is_server_ready())

# Per-model readiness (specific model is loaded)
print(client.is_model_ready("resnet50_trt"))

# Model metadata (tensor names, shapes, types)
metadata = client.get_model_metadata("resnet50_trt")
```

## 12. GPU memory management: the pool architecture

Triton pre-allocates GPU memory at server startup rather than dynamically allocating per request. This design eliminates `cudaMalloc` overhead from the critical inference path and prevents memory fragmentation that degrades throughput over time as thousands of allocations and frees leave the GPU heap in a fragmented state.

The two key pools are controlled by server startup flags:

**CUDA Memory Pool** (`--cuda-memory-pool-byte-size=<gpu_id>:<bytes>`):

Pre-allocates a contiguous block of device memory for inference I/O buffers. When a request arrives, Triton carves input and output tensors from this pool using a fast slab allocator rather than calling `cudaMalloc`. The pool is per-GPU and shared across all models on that GPU.

Sizing rule of thumb: allocate at minimum `max_batch_size × input_tensor_bytes × 2 × num_models` (the factor of 2 is for double-buffering concurrent prefetch). For ResNet-50 at batch 64: `64 × 3 × 224 × 224 × 4 bytes = 38.5MB`. For 4 instances double-buffered: approximately 310MB. The default pool (100MB per GPU) is too small for multi-model deployments — always set this explicitly and confirm via `nv_gpu_memory_used_bytes` in Prometheus.

**Pinned Memory Pool** (`--pinned-memory-pool-byte-size=<bytes>`):

Pinned (page-locked) CPU memory enables zero-copy host-to-device DMA via the PCIe bus. Without it, every input tensor transfer requires: `cudaMallocHost` (pin the buffer) → `cudaMemcpy` (transfer) → `cudaFreeHost` (unpin) — three CUDA API calls per request. The pinned pool pre-allocates a large page-locked region and services requests from it, reducing this to a single `cudaMemcpy`. For a 1-Gbps NIC receiving images at 125MB/s throughput, keeping 2–4GB of pinned memory ensures the host-side buffer never becomes the bottleneck.

**Model weight loading**: model weights are loaded into GPU memory when the model loads and stay resident unless explicitly unloaded. You control startup parallelism with `--model-load-thread-count N` — loading 4 models in parallel versus sequentially cuts startup time significantly for large model repositories. The rate limit mode (`--rate-limit`) adds token-based request throttling at the server level, independent of per-model batching.

**CUDA unified memory**: Triton supports CUDA Unified Memory (`--cuda-unified-memory-pool-byte-size`) for systems where CPU and GPU share physical memory (e.g., Jetson AGX Orin). Unified memory eliminates the PCIe copy entirely — both CPU and GPU can read and write the same physical address. On discrete GPU workstations (A100, H100), unified memory still requires PCIe transfers and is slower than the pinned pool approach for high-throughput serving. Do not use unified memory on discrete GPU servers.

**Memory fragmentation over time**: even with pre-allocated pools, long-running Triton deployments (weeks without restart) can accumulate fragmentation inside the slab allocator if request sizes vary widely. Monitor `nv_gpu_memory_used_bytes` over time — a steady upward drift without a corresponding load increase is fragmentation. The fix is a controlled pod restart during a low-traffic window, using Kubernetes rolling updates to maintain serving capacity.

## 13. Prometheus metrics: the full observability stack

Triton exposes a Prometheus-compatible endpoint at `:8002/metrics`. The metrics are counters and histograms that cover every critical SLO dimension, including per-phase latency breakdowns that are unique to Triton.

**Core inference metrics**:

| Metric | Type | What it measures |
|---|---|---|
| `nv_inference_request_success` | Counter | Total successful inference requests |
| `nv_inference_request_failure` | Counter | Failed requests (timeout, OOM, bad input) |
| `nv_inference_count` | Counter | Total inferences (requests × batch_size filled) |
| `nv_inference_exec_count` | Counter | Backend kernel launches |
| `nv_inference_request_duration_us` | Counter | Total end-to-end request latency |
| `nv_inference_queue_duration_us` | Counter | Time waiting in the scheduler queue |
| `nv_inference_compute_input_duration_us` | Counter | Host-to-device tensor copy time |
| `nv_inference_compute_infer_duration_us` | Counter | Actual backend compute time |
| `nv_inference_compute_output_duration_us` | Counter | Device-to-host tensor copy time |
| `nv_inference_pending_request_count` | Gauge | Current queue depth per model |
| `nv_gpu_memory_total_bytes` | Gauge | Total GPU memory |
| `nv_gpu_memory_used_bytes` | Gauge | Current GPU memory in use |
| `nv_gpu_utilization` | Gauge | GPU SM utilization percent |

Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: triton
    static_configs:
      - targets: ['triton-service:8002']
    scrape_interval: 15s
    metrics_path: /metrics
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
```

**Derived PromQL for Grafana panels**:

```promql
# Request throughput (req/s, 1-minute rolling window)
rate(nv_inference_request_success{model="resnet50_trt"}[1m])

# p99 queue latency in milliseconds
histogram_quantile(0.99,
  rate(nv_inference_queue_duration_us_bucket{model="resnet50_trt"}[5m])
) / 1000

# Batch efficiency: average batch size at dispatch
rate(nv_inference_count[1m]) / rate(nv_inference_exec_count[1m])

# Backend compute utilization: fraction of request time spent on actual compute
rate(nv_inference_compute_infer_duration_us[1m]) /
  rate(nv_inference_request_duration_us[1m])

# GPU memory pressure
nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes * 100

# Error rate
rate(nv_inference_request_failure[5m]) /
  (rate(nv_inference_request_success[5m]) +
   rate(nv_inference_request_failure[5m])) * 100
```

![Prometheus metrics flow: Triton to Grafana with alerting](/imgs/blogs/triton-inference-server-deep-dive-8.png)

**Grafana alert rule** for p99 SLA breach:

```yaml
groups:
  - name: triton-slo
    rules:
      - alert: TritonP99LatencyHigh
        expr: |
          histogram_quantile(0.99,
            rate(nv_inference_queue_duration_us_bucket[5m])
          ) / 1000 > 50
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Triton p99 queue latency above 50ms SLA"
          description: "Model {{ $labels.model }} p99 = {{ $value }}ms"
      - alert: TritonQueueDepthHigh
        expr: nv_inference_pending_request_count > 200
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Triton queue depth exceeds 200 requests"
          description: "Model {{ $labels.model }} queue depth = {{ $value }}"
```

**The most important derived metric** is batch efficiency: `rate(nv_inference_count) / rate(nv_inference_exec_count)`. Target value: 60–80% of your `preferred_batch_size`. If batch efficiency is consistently below 4 and GPU utilization is also low, you are under-batching — raise `max_queue_delay_microseconds`. If GPU utilization is above 90% and p99 is climbing, you are saturated — add more model instances or scale the pod horizontally.

## 14. Deploying Triton on Kubernetes

Production Triton deployments run on Kubernetes with GPU node selectors and resource limits. Here is a complete Deployment and Service manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference-server
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8002"
        prometheus.io/path: "/metrics"
    spec:
      nodeSelector:
        nvidia.com/gpu.product: "A100-SXM4-40GB"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: triton
          image: nvcr.io/nvidia/tritonserver:24.05-py3
          args:
            - tritonserver
            - --model-repository=s3://ml-models/prod-repo
            - --strict-model-config=true
            - --log-verbose=0
            - --cuda-memory-pool-byte-size=0:12000000000
            - --pinned-memory-pool-byte-size=4000000000
            - --model-control-mode=explicit
          ports:
            - containerPort: 8000   # HTTP
            - containerPort: 8001   # gRPC
            - containerPort: 8002   # Prometheus
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "48Gi"
              cpu: "8"
            requests:
              nvidia.com/gpu: 1
              memory: "32Gi"
              cpu: "4"
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 20
            failureThreshold: 3
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi
# --- Service manifest below ---
apiVersion: v1
kind: Service
metadata:
  name: triton-service
  namespace: ml-serving
spec:
  selector:
    app: triton
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: grpc
      port: 8001
      targetPort: 8001
    - name: metrics
      port: 8002
      targetPort: 8002
  type: ClusterIP
```

**Horizontal Pod Autoscaler on queue depth**: scale Triton pods based on the Prometheus metric `nv_inference_pending_request_count`:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: triton-scaler
  namespace: ml-serving
spec:
  scaleTargetRef:
    name: triton-inference-server
  minReplicaCount: 2
  maxReplicaCount: 8
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: nv_inference_pending_request_count
        query: |
          max(nv_inference_pending_request_count{model="resnet50_trt"})
        threshold: "50"
```

This KEDA ScaledObject triggers a scale-out when the queue depth for ResNet exceeds 50 requests, adding Triton pod replicas (each with a full A100) until the queue drains. The `minReplicaCount: 2` ensures no cold-start latency spike during traffic ramps.

**Rolling updates**: when you push a new model version to S3, you trigger a rolling update by patching the Deployment. Use an init container pattern to pre-load models before the readiness probe fires:

```yaml
initContainers:
  - name: model-loader
    image: nvcr.io/nvidia/tritonserver:24.05-py3
    command:
      - sh
      - -c
      - |
        curl -X POST "localhost:8000/v2/repository/models/resnet50_trt/load" &&
        curl -X POST "localhost:8000/v2/repository/models/bert_base_onnx/load"
```

## 15. Rate limiting, request priority, and model warmup

Production Triton deployments need more than throughput tuning — they need protection against runaway clients, priority queuing for SLA tiers, and warm startup behavior to avoid cold-start spikes.

**Server-level rate limiting**: Triton's `--rate-limit` flag enables a global token-bucket rate limiter across the entire server. This controls how many requests per second the server will accept regardless of model-level queue depth:

```bash
tritonserver \
  --rate-limit=exec_count \
  --rate-limit-resource=R1,10 \
  --rate-limit-resource=R2,5 \
  --model-repository=/models
```

Rate limit resources are user-defined tokens that models consume on each execution. In `config.pbtxt`, a model can declare its resource consumption:

```protobuf
name: "resnet50_trt"
platform: "tensorrt_plan"
max_batch_size: 64

rate_limit_resources [
  { name: "R1", count: 1 }
]
```

This declares that each ResNet execution consumes one R1 token. With `--rate-limit-resource=R1,10`, Triton allows at most 10 concurrent ResNet executions at any moment — independent of the dynamic batching queue. This is useful when you have a GPU-intensive post-processing step that cannot be batched and you need to prevent it from starving other models.

**Request priorities**: Triton supports multiple priority levels via the `priority` field in the inference request. Requests with lower priority numbers are served first:

```python
# High-priority request (priority=1, served first)
result = client.infer(
    model_name="resnet50_trt",
    inputs=inputs,
    outputs=outputs,
    priority=1    # lower number = higher priority
)

# Background batch request (priority=4)
result = client.infer(
    model_name="resnet50_trt",
    inputs=inputs,
    outputs=outputs,
    priority=4
)
```

Priority is enforced within the dynamic batching scheduler: when forming a batch, Triton pulls the highest-priority requests from the queue first. This enables a two-tier serving architecture: interactive user requests at priority 1 share the same model instance with background analytics jobs at priority 4, without needing separate Triton deployments or GPU resources.

**Request timeouts**: clients can set a per-request timeout via the gRPC deadline mechanism. On the server side, Triton enforces this via `timeout_microseconds` in `config.pbtxt`:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 32, 64 ]
  max_queue_delay_microseconds: 5000
  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 50000    # 50ms queue timeout
    allow_timeout_override: true
  }
}
```

With `timeout_action: REJECT`, requests that have waited longer than 50ms in the queue are rejected with a gRPC DEADLINE_EXCEEDED error rather than being processed late. This is critical for latency-SLA services: a request that would be served at 100ms p99 is worse than a fast rejection that lets the client retry or fall back — keeping your p99 metric clean.

**Model warmup**: TensorRT engines JIT-compile CUDA kernels on first execution, causing a dramatic latency spike for the first N requests after startup. Triton's model warmup feature runs dummy requests through the model at startup before the readiness probe fires:

```protobuf
model_warmup [
  {
    name: "warmup_batch32"
    batch_size: 32
    inputs {
      key: "input"
      value {
        data_type: TYPE_FP32
        dims: [3, 224, 224]
        zero_data: true    # fill with zeros
      }
    }
  }
]
```

Add this block to your ResNet `config.pbtxt`. Triton runs this warmup inference (and discards the result) before marking the model as ready. For BERT with three inputs, add a warmup block for each input tensor. The warmup typically adds 5–15 seconds to model load time but eliminates the first-request latency spike that would otherwise fire your SLO alert.

**Graceful shutdown and connection draining**: in Kubernetes rolling updates, Triton needs to drain in-flight requests before the pod terminates. Configure a `preStop` hook in the pod spec:

```yaml
lifecycle:
  preStop:
    exec:
      command:
        - sh
        - -c
        - "sleep 10"   # allow 10s for in-flight requests to complete
```

Combined with a `terminationGracePeriodSeconds: 30` on the pod spec, this gives Triton 30 seconds to finish serving before the kubelet sends SIGKILL. For long-running batch requests (BERT at batch 32 with 128-token sequences takes ~4ms), even 5 seconds is more than sufficient.

## 16. Debugging common Triton failures

Production Triton deployments fail in a small set of predictable patterns. Knowing these saves hours of debugging.

**Model fails to load — `config.pbtxt` dimension mismatch**: the most common error. You see:
```
Failed to load model "resnet50_trt" version 1: Internal: Failed to load TensorRT plan from file
```
Usually caused by: (1) `dims` in config includes the batch dimension (should NOT), (2) `max_batch_size` exceeds the TRT engine's compiled max, or (3) `data_type` mismatch (config says `TYPE_FP32`, engine was built for `TYPE_FP16`). Fix: enable `--strict-model-config=false` temporarily to let Triton auto-infer dims, then compare the auto-inferred config against your handwritten one.

**OOM on model load**: Triton loads all models at startup and each instance allocates VRAM. You see:
```
RuntimeError: CUDA error: out of memory
```
Debug with `nvidia-smi` at startup: watch VRAM usage increment per model. Reduce `instance_group count`, switch FP32 models to FP16, or use MIG partitioning. Set `--cuda-memory-pool-byte-size` explicitly — the default pool consumes VRAM even before models load.

**Dynamic batching never fires — batch size stays at 1**: usually caused by `max_queue_delay_microseconds` being too small (e.g., `100` µs at 50 req/s arrival rate — the queue cannot accumulate 32 requests in 100µs). Calculate the theoretical delay: at 50 req/s and preferred batch 32, you need $32/50 = 640$ms of delay for full batches. This is almost always too high for interactive use — set preferred_batch_size to 4–8 instead and keep the delay at 5–10ms.

**Python backend hangs**: the most frustrating failure. The `execute()` method is called with a list of requests, and if any exception is not caught, the model instance hangs indefinitely (Triton's Python backend worker does not have a per-call timeout). Fix: wrap the entire `execute()` body in a try/except:

```python
def execute(self, requests):
    responses = []
    for request in requests:
        try:
            # ... processing ...
            responses.append(pb_utils.InferenceResponse(output_tensors=[...]))
        except Exception as e:
            responses.append(pb_utils.InferenceResponse(
                error=pb_utils.TritonError(f"Preprocessing failed: {e}")
            ))
    return responses
```

**gRPC channel errors at high concurrency**: when sending 10,000+ requests per second, the default gRPC channel hits connection limits. Fix: create a pool of clients and distribute requests across them:

```python
import itertools
import tritonclient.grpc as grpcclient

# Create a pool of 8 connections to spread load
client_pool = [
    grpcclient.InferenceServerClient("localhost:8001")
    for _ in range(8)
]
pool_cycle = itertools.cycle(client_pool)

def get_client():
    return next(pool_cycle)
```

For very high throughput, switch to the gRPC streaming API (`async_stream_infer`) which amortizes connection setup overhead across thousands of requests on a single persistent stream.

**Ensemble step fails silently**: if a step in an ensemble fails, Triton propagates the error back to the client as a gRPC error code, but the error message may not clearly identify which step failed. Enable verbose logging (`--log-verbose=2`) and watch for lines like:
```
[tritonserver] Error for model "preprocess_python": ...
```
The log will identify the step name and the specific error. Add explicit error handling in your Python backend's `execute()` method and include the step name in error messages.

These failure patterns collectively cover over 90% of production Triton issues. The rest are typically GPU driver mismatches (container CUDA version vs host driver version — always match the NGC container tag to your driver), or NFS/S3 access permissions on the model repository path.

## 17. Case studies and benchmarks

### ResNet-50 throughput on A100: ONNX vs TRT

From NVIDIA's published benchmarks for ResNet-50v1.5, A100 SXM 80GB, batch size 64, single instance:

| Backend | Throughput (images/s) | p99 latency | One-time build cost |
|---|---|---|---|
| ONNX Runtime (CUDA EP) | 18,200 | 3.7ms | Under 1 min |
| TensorRT FP16 | 45,100 | 1.4ms | 12 min |
| TensorRT INT8 (calibrated) | 61,800 | 1.1ms | 18 min + calibration dataset |

Source: NVIDIA MLPerf Inference v3.1 closed division results, ResNet-50 offline scenario.

TRT INT8 delivers 3.4× the throughput of ONNX Runtime at 73% lower latency — all for a one-time 18-minute build investment. For a production system serving ten million requests per day, a 3.4× throughput improvement means running one GPU instead of three at equivalent cost. The INT8 accuracy drop on ImageNet top-1 is approximately 0.3% — acceptable for most production classification tasks.

### BERT-base serving: Triton vs TorchServe

From NVIDIA's TensorRT optimization blog (2023), BERT-base uncased, batch size 32, sequence length 128, T4 16GB:

| Server | Backend | Throughput | p99 latency |
|---|---|---|---|
| TorchServe | PyTorch eager | 180 seq/s | 187ms |
| Triton | ONNX Runtime (CUDA EP) | 420 seq/s | 78ms |
| Triton | TRT FP16 | 890 seq/s | 36ms |
| Triton | TRT INT8 | 1,240 seq/s | 25ms |

Triton with TRT INT8 is 6.9× faster than TorchServe eager mode on the same hardware. The difference comes from three compounding factors: TRT kernel fusion eliminates intermediate tensor materializations in the attention layers, dynamic batching fills GPU SMs that TorchServe's one-request-per-thread model leaves idle, and Triton's multi-instance execution overlaps I/O and compute. The INT8 compression is especially effective on BERT because the dominant operations — general matrix multiplications in the attention blocks — benefit disproportionately from INT8 tensor core acceleration on NVIDIA Ampere and later architectures. On A100, INT8 tensor cores deliver 2× the throughput of FP16 for the same FLOP count, which explains why the TRT INT8 improvement (from 890 to 1,240 seq/s) is additive on top of the FP16 improvement.

### Multi-model co-hosting cost savings

A practical production example: an e-commerce platform serves ResNet-50 for visual search (3,000 req/s peak) and BERT for query understanding (800 req/s peak) as separate services on separate `p3.2xlarge` V100 instances (\$3.06/hr on-demand). Total: \$6.12/hr for two GPUs, each running at 35–40% utilization.

After migration to Triton on a single `p3.2xlarge`, co-hosting both models with the instance group tuning described earlier: one GPU at 82% utilization, \$3.06/hr. Cost savings: \$3.06/hr = \$26,800/year, with better GPU utilization and lower response variance due to reduced queue contention.

#### Worked example: A100 multi-model deployment sizing

**Goal**: Serve ResNet-50 (computer vision) + BERT-base (NLP) at 2,000 req/s for ResNet and 500 req/s for BERT, p99 under 20ms for both, on a single A100 40GB.

**Step 1: single-instance throughput capacity**
From `perf_analyzer` sweeps:
- ResNet TRT FP16, 1 instance, concurrency 8: ~1,250 infer/s, p99 7ms
- BERT ONNX FP32, 1 instance, concurrency 4: ~480 infer/s, p99 12ms

**Step 2: instance count to meet QPS targets**
- ResNet: 2,000 / 1,250 = 1.6 → 2 instances (capacity: 2,500/s, headroom 25%)
- BERT: 500 / 480 = 1.04 → 1 instance with ample margin

**Step 3: VRAM verification**
- ResNet TRT FP16 × 2: 2 × 168MB = 336MB
- BERT ONNX FP32 × 1: 4.8GB (weights + activations at batch 32)
- CUDA I/O pool: 2GB
- Context overhead: ~1GB
- Total: approximately 8.1GB — well within the 40GB budget

**Step 4: final config.pbtxt knobs**

```protobuf
# resnet50_trt/config.pbtxt
instance_group [{ kind: KIND_GPU, count: 2, gpus: [0] }]
dynamic_batching {
  preferred_batch_size: [32, 64]
  max_queue_delay_microseconds: 5000
}

# bert_base_onnx/config.pbtxt
instance_group [{ kind: KIND_GPU, count: 1, gpus: [0] }]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 8000
}
```

**Step 5: cost model** on one `p3.2xlarge` V100 (\$3.06/hr spot, ~70% availability):
- Cost per 1M ResNet inferences: \$3.06 / (2,500 × 3,600) × 1,000,000 ≈ \$0.34
- Cost per 1M BERT inferences: \$3.06 / (480 × 3,600) × 1,000,000 ≈ \$1.77

Versus two separate instances (one for each model):
- ResNet only: \$3.06/hr / 2,500 × 1M / 3,600 = \$0.34/1M (same GPU spec)
- BERT only: \$3.06/hr / 480 × 1M / 3,600 = \$1.77/1M

The cost-per-inference is identical — you are just running both on one GPU instead of two, halving the fixed cost. This is the core economic argument for Triton's multi-model hosting.

## 18. When Triton wins — and when it does not

**Triton wins when**:

- You have multiple models (two or more) that need to share one or more GPUs. Triton's instance group and scheduler architecture is purpose-built for this and has no equivalent in TorchServe or BentoML.
- Your workload is CV, audio, tabular ML, or embedding generation — fixed-shape inputs work perfectly with TensorRT and ONNX Runtime.
- You need pipeline ensembles — Triton's ensemble scheduling is the only production-grade framework-native solution for multi-model DAG inference without a custom microservice orchestrator.
- You need the full TensorRT optimization chain (INT8 calibration, layer fusion, FP8 on H100) accessible through a standard serving API.
- You have a mixed-framework fleet — some PyTorch models, some TensorFlow SavedModels, some ONNX exports — and want one uniform API in front of all of them.
- Observability depth matters: Triton's Prometheus metrics expose per-phase latency (queue, compute_input, compute_infer, compute_output) out of the box. No other serving framework provides this level of granularity without custom instrumentation.

**Triton loses when**:

- You are serving LLMs above approximately 20 concurrent requests. PagedAttention, continuous batching, and per-request KV cache memory management are not natively supported in Triton's scheduler. The vLLM backend is experimental and bypasses Triton's scheduler entirely for LLM workloads. Use vLLM or TGI directly. See [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) for the full explanation.
- You have a single simple model that will never share a GPU with others. TorchServe ([TorchServe deep dive](/blog/machine-learning/model-serving/torchserve-deep-dive)) is simpler to configure, has a cleaner model archive format, and a management API that supports dynamic loading without protobuf configs.
- Your team is Python-only and has no appetite for protobuf configs and `config.pbtxt` debugging. Misconfigured tensor shapes in `config.pbtxt` cause cryptic CUDA errors at runtime that are hard to diagnose without experience.
- You need dynamic input shapes at high throughput — variable-length sequences of widely varying lengths, variable-resolution images. Triton supports dynamic shapes via TRT optimization profiles and ONNX symbolic dims, but performance degrades compared to fixed shapes, and the configuration complexity is substantial.
- You are deploying to CPU-only environments or edge hardware. Triton's GPU-focused design makes it heavier than needed for CPU serving; ONNX Runtime standalone or OpenVINO Model Server are better fits.

| Scenario | Recommended tool | Triton verdict |
|---|---|---|
| ResNet + BERT + preprocessing DAG on GPU | **Triton** | Purpose-built for this exact case |
| Single LLM 7B–70B, interactive QPS | vLLM | Triton lacks KV cache management |
| Single PyTorch model, small team | TorchServe | Simpler config, better hot-reload |
| A/B testing two PyTorch model variants | Ray Serve | Native traffic split and autoscaling |
| Edge device, CPU only | ONNX Runtime standalone | Triton overhead not justified |
| Large mixed GPU+CPU fleet, cloud-agnostic | BentoML | Hardware-neutral wrapper layer |

**The migration decision**: if you currently run TorchServe for a single ResNet model and plan to add a second model (a text encoder, a fraud classifier, anything), that is the moment to migrate to Triton. The migration cost — rewriting from `.mar` to ONNX or TRT + `config.pbtxt`, switching from TorchServe's REST-JSON API to Triton's gRPC or HTTP/KFServing API — is worth doing once. Doing it twice (TorchServe → Triton when you add the second model, then back to vLLM when you add an LLM) is more disruptive.

**When to accept Triton's operational cost**: Triton is genuinely complex. The `config.pbtxt` protobuf format is verbose. Dynamic batching semantics differ between backend types. Python backend debugging is difficult. TRT `.plan` files are not portable. These costs are real and should not be dismissed. Accept them when the throughput gains, co-hosting economics, or observability requirements are worth the operational investment. For teams with fewer than three engineers managing model serving, a managed service (AWS SageMaker, Google Vertex AI) that wraps Triton internally may deliver most of the benefit with a fraction of the operational burden.

## 19. Key takeaways

1. The `config.pbtxt` is the single most important file in a Triton deployment — every performance and reliability property is expressed here. Treat it with the same rigor as application code and version-control it alongside model artifacts.

2. `max_queue_delay_microseconds` and `preferred_batch_size` are your primary throughput-latency knobs. The delay budget follows the formula $B/\lambda$ for Poisson arrivals; the preferred batch size should target 80–90% GPU SM utilization from a `perf_analyzer` sweep.

3. Use `instance_group count: N` to fill GPU SM idle gaps. For CV models on A100, four concurrent instances is a common sweet spot. Always verify with `perf_analyzer` — more instances is not always better when activation memory pressure limits the GPU's ability to overlap streams.

4. TensorRT delivers 3–6× throughput over ONNX Runtime on standard CV workloads. Reserve the 10–30 minute build time for production deployments; use ONNX ORT during development. Remember that `.plan` files are architecture-specific: A100 plans fail on T4 and vice versa.

5. Ensemble models eliminate client-side orchestration round-trips. For co-located pipelines the overhead is under 1ms; for cross-datacenter deployments, ensembles can eliminate 50–200ms of avoidable latency per hop in a multi-step pipeline.

6. The Python backend is an escape hatch, not a performance path. Use it for preprocessing and postprocessing steps that contain Python-only logic; keep the model execution step on TRT or ONNX.

7. Monitor batch efficiency (`rate(nv_inference_count) / rate(nv_inference_exec_count)`) and queue depth (`nv_inference_pending_request_count`) as leading indicators. Batch efficiency below 4 with low GPU utilization means under-batching — raise `max_queue_delay_microseconds`. Growing queue depth with high GPU utilization means the backend cannot keep up — add instances or scale pods.

8. Do not use Triton for LLM serving above approximately 20 QPS without the vLLM backend. The scheduler does not implement KV cache memory management, continuous batching, or PagedAttention.

9. Pre-allocate the CUDA memory pool (`--cuda-memory-pool-byte-size`) and pinned memory pool (`--pinned-memory-pool-byte-size`) explicitly at server startup. The defaults are undersized for multi-model deployments and cause latency spikes under burst traffic when CUDA malloc is called on the hot path.

10. The standard Triton tuning workflow: launch the server, run `perf_analyzer --concurrency-range 1:16`, identify the knee where p99 starts inflecting, set `preferred_batch_size` to target 80% throughput below the knee, then push to production with Prometheus alerts on queue depth and p99 latency.

11. Model warmup via the `model_warmup` block in `config.pbtxt` eliminates TRT JIT compilation on the first production request. Always include a warmup step for TRT models; the 5–15 second startup cost is invisible to users but saves a 200–400ms first-request spike that would breach any interactive SLA.

12. Use request priorities (the `priority` field in the gRPC inference call) to share a single Triton deployment between interactive user-facing traffic (priority 1) and background batch analytics jobs (priority 4). This is more cost-efficient than separate Triton deployments per workload class when peak times do not overlap.

## 20. The production Triton tuning checklist

Before a Triton deployment reaches production, walk through this checklist. Each item maps to a failure mode seen in real deployments.

**Repository and config:**
- [ ] All `config.pbtxt` files are under version control alongside model artifacts
- [ ] `max_batch_size` matches the TRT engine's compiled max (or is set conservatively for ONNX)
- [ ] `dims` fields do NOT include the batch dimension
- [ ] `instance_group` count was validated with a `perf_analyzer` sweep, not assumed
- [ ] Dynamic batching preferred_batch_size list targets 80% of the throughput knee
- [ ] `model_warmup` blocks are present for all TRT models

**Server flags:**
- [ ] `--cuda-memory-pool-byte-size` is set explicitly per GPU (not relying on 100MB default)
- [ ] `--pinned-memory-pool-byte-size` is at least 2× the peak request throughput in MB/s × 0.020s
- [ ] `--strict-model-config=true` is set for production (auto-inference disabled)
- [ ] `--model-control-mode=explicit` so accidental repo changes don't auto-load

**Kubernetes:**
- [ ] Readiness probe uses `/v2/health/ready` (not `/v2/health/live`) to delay traffic until all models are loaded
- [ ] Pod `terminationGracePeriodSeconds` is at least 30 seconds + max expected request service time
- [ ] KEDA or HPA is wired to `nv_inference_pending_request_count` for queue-depth-based scaling
- [ ] GPU resource limits are set to exactly 1 GPU per pod (never fractional without MIG)

**Observability:**
- [ ] Prometheus scrape is configured for port 8002
- [ ] Batch efficiency alert fires below 4 × preferred_batch_size at expected QPS
- [ ] p99 queue latency alert fires at 70% of the SLA budget (not 100% — alerting at the limit leaves no response time)
- [ ] Queue depth alert fires early enough to allow scale-out before SLA breach

**Load testing:**
- [ ] `perf_analyzer --concurrency-range 1:32` completed on production hardware (not dev laptop)
- [ ] Load test included a burst scenario at 5× average QPS for 60 seconds to validate autoscaler response
- [ ] Ensemble load test (`perf_analyzer -m classify_ensemble`) completed to validate pipeline latency budget

This checklist has prevented more 3 AM pages than any other single artifact in Triton deployments. Print it, version-control it, and require its completion before any Triton service reaches production.

## 21. Further reading

- **NVIDIA Triton Inference Server documentation**: [https://docs.nvidia.com/deeplearning/triton-inference-server/](https://docs.nvidia.com/deeplearning/triton-inference-server/) — authoritative source for all `config.pbtxt` fields, backend APIs, and performance tuning guides.
- **Triton GitHub tutorials**: [https://github.com/triton-inference-server/tutorials](https://github.com/triton-inference-server/tutorials) — end-to-end examples for every backend including Python backend patterns, ensemble configurations, and Kubernetes deployment manifests.
- **NVIDIA TensorRT documentation**: [https://docs.nvidia.com/deeplearning/tensorrt/](https://docs.nvidia.com/deeplearning/tensorrt/) — TRT builder API, INT8 calibration workflow, engine optimization reference, and the `trtexec` flag reference.
- **MLPerf Inference benchmark results**: [https://mlcommons.org/en/inference-edge-21/](https://mlcommons.org/en/inference-edge-21/) — published throughput and latency numbers for ResNet, BERT, and other models across GPU hardware tiers under MLPerf's standardized benchmark methodology.
- [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — series introduction covering the SLO triangle and the full serving stack, with framework comparisons.
- [TorchServe deep dive](/blog/machine-learning/model-serving/torchserve-deep-dive) — companion post on TorchServe's single-model strengths versus Triton's multi-model architecture.
- [ONNX Runtime for serving](/blog/machine-learning/model-serving/onnx-runtime-for-serving) — deep dive on ORT execution providers, session options, and optimization graph transforms.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — series capstone: full decision tree from trained model to production deployment, summarizing when each framework wins.
- [Choosing your serving stack](/blog/machine-learning/model-serving/choosing-your-serving-stack) — decision matrix post: Triton vs Ray Serve vs TorchServe vs vLLM, with workload-specific recommendations and migration paths between frameworks.
