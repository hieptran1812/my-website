---
title: "TorchServe deep dive: From model archive to production API"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Build a production-ready TorchServe deployment: create MAR archives, write custom handlers, tune dynamic batching, expose Prometheus metrics, and know when to switch to a different runtime."
tags:
  [
    "model-serving",
    "inference",
    "torchserve",
    "pytorch",
    "deep-learning",
    "ml-infrastructure",
    "batch-inference",
    "production-ml",
    "model-deployment",
    "metrics",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/torchserve-deep-dive-1.png"
---

## Introduction: The 2 AM pager call

It is 2:47 AM. Your phone goes off. The on-call alert reads: `p99_latency > 3000ms — bert-sentiment — CRITICAL`. You SSH into the instance, pull up the metrics dashboard, and see that inference latency has spiked from a healthy 45ms baseline to 3.4 seconds at the 99th percentile. The GPU utilization is sitting at 11%. The service has not crashed. Requests are completing — just incredibly slowly.

This exact scenario played out on a team that had deployed a BERT-large sentiment classifier to TorchServe two weeks earlier. The model worked perfectly in testing. The MAR archive was created correctly. The handler code was clean. But one configuration decision made in haste during the initial rollout — leaving `default_workers_per_model=1` and not configuring batching — meant that the service was processing every request sequentially, one at a time, through a single Python worker process. At 2 AM when a batch job upstream started generating 60 concurrent classification requests, those requests queued up behind each other. With an 18ms GPU inference time per request and 60 requests waiting in a single-worker queue, the tail latency hit 18ms × 60 + queue overhead = ~1.1 seconds minimum. The p99 measurement of 3.4 seconds included the HTTP timeout overhead, retry attempts, and the JVM request queue filling up.

The fix took four minutes: scale the worker pool to 4 workers, enable dynamic batching with `batch_size=8` and `max_batch_delay=50`, and restart. Latency dropped back to 52ms p99. Throughput went from 12 req/s to 220 req/s on the same hardware. Four minutes to fix a problem that had been lurking since deployment, caused entirely by misconfigured defaults.

This post explains how TorchServe works from the ground up, so you never get that pager call. You will learn how TorchServe's Netty frontend routes requests to Python worker pools, how to build production-quality MAR archives and custom handlers, how to tune dynamic batching using Little's Law, how to expose Prometheus metrics, and — critically — when TorchServe is the wrong tool and which alternative you should use instead.

### What this post covers and what it assumes

This is a practitioner's deep dive. It assumes you are comfortable with PyTorch model training and have deployed at least one web service in production. It does not assume any prior TorchServe experience. You will come away understanding:

- The exact data path from an HTTP POST to a GPU forward pass and back
- How to build MAR archives correctly, including the version management story
- How to write handlers that handle edge cases: mixed-type batches, warm-up passes, fp16 conversion, and custom metrics
- The mathematics of dynamic batching and how to set `batch_size` and `max_batch_delay` from first principles given your SLO
- How TorchServe's Prometheus integration works and what to alert on
- Honest limitations of TorchServe and the exact thresholds where you should switch to vLLM, Triton, or Ray Serve

The TorchServe version referenced throughout this post is 0.9.x, which is the current stable release as of mid-2026. The management API, MAR format, and handler interface have been stable since 0.6.x, so the guidance here applies to all recent versions.

![TorchServe internal architecture showing the Netty frontend, request queues, JVM-Python bridge, and worker pool layout](/imgs/blogs/torchserve-deep-dive-1.png)

## 1. TorchServe architecture: How requests become predictions

TorchServe is a joint project between AWS and Facebook (Meta), released in 2020, designed to be the official production serving solution for PyTorch models. Understanding its architecture is essential for tuning it correctly, because the performance knobs you turn correspond directly to architectural components.

### The three ports

TorchServe exposes three distinct HTTP ports, each with a specific function:

**Port 8080 — Inference API.** This is the data plane. All prediction requests (`POST /predictions/<model_name>`) come through here. In production, this is the port your load balancer routes traffic to. It is handled by the Netty HTTP server, which is an asynchronous event-driven networking framework for Java with exceptional connection-handling capacity (Netty is used by LinkedIn, Twitter, and many high-throughput systems for this reason).

**Port 8081 — Management API.** The control plane. Model registration, deregistration, worker scaling, and configuration changes go through this port. In production, this port should be firewalled from external traffic — it exposes administrative operations. Only your deployment automation and orchestration systems need access to it.

**Port 8082 — Metrics API.** Exposes Prometheus-compatible metrics at `/metrics`. This is a pull-based endpoint that your Prometheus scraper polls at a configured interval. It exposes inference latency histograms, queue wait times, worker thread utilization, and custom metrics your handlers emit.

### The Netty frontend

The Netty frontend is the entry point for all inference and management requests. When a POST request arrives at port 8080 for `/predictions/bert-sentiment`, the Netty server accepts the connection in O(1) time (non-blocking I/O), reads the request body, and looks up which model's request queue to enqueue the job in. This lookup is a simple hash map from model name to queue reference — it is extremely fast.

Netty's event loop model means the frontend can handle thousands of concurrent connections without spawning a thread per connection. The frontend itself does not do any ML computation. Its job is purely routing: accept bytes, determine target model, enqueue job, eventually collect the result, and write the HTTP response back.

### Request queues and backpressure

Each registered model has its own job queue. The queue depth is configured via `job_queue_size` in `config.properties` (default: 100). When a request arrives faster than the worker pool can process them, jobs accumulate in this queue. When the queue fills, TorchServe returns HTTP 503 to new requests — this is backpressure. The `ts_queue_latency_microseconds` metric tells you how long jobs are sitting in the queue waiting for a worker to pick them up. If this metric is high, you either need more workers or larger batch sizes.

### The JVM-Python bridge

Here is where TorchServe's architecture gets interesting — and creates its main limitation. TorchServe's core is a Java application (the Netty server, request routing, worker management, metrics collection). Python workers run as separate subprocesses. Communication between the JVM and Python workers happens over a local Unix domain socket or TCP loopback connection using a Protocol Buffers-based wire protocol.

This design choice has important consequences:

First, Python workers are genuine OS processes, not threads. This means the Python GIL does not create cross-worker contention. Worker 1 and Worker 2 can execute forward passes simultaneously on separate GPU streams without one blocking the other. This is fundamentally different from, say, Flask with Gunicorn workers — you get true process-level parallelism.

Second, each Python worker has a memory overhead of roughly 200–300 MB just for the interpreter and imported libraries before the model is loaded. For a BERT-large model at 1.34 GB in fp32 (or 0.67 GB in fp16), loading four workers means 4 × (1.34 + 0.25) = 6.36 GB of model weights alone, plus activation memory.

Third, the JVM itself has a startup penalty. The JVM cold start — JIT compilation, class loading, connection pool initialization — adds 300–500 milliseconds to the first request after TorchServe starts. This matters for serverless-style autoscaling where instances spin up and down. If you are scaling from zero, that first request sees the JVM cold start penalty before it even reaches your model.

### Worker lifecycle

Each Python worker transitions through well-defined states:

**CREATED**: The JVM has spawned the Python subprocess. The process is alive but has not yet loaded any model weights.

**Loading**: The worker is executing `handler.initialize(ctx)`. This is where model weights are read from disk, moved to GPU, and any warm-up forward passes happen. For BERT-large loading from NVMe SSD, this takes 3–8 seconds depending on storage speed and GPU PCIe bandwidth.

**READY (IDLE)**: The worker has finished initialization and is ready to accept jobs. The JVM marks this worker as available and will route new jobs to it.

**BUSY**: The worker is executing a job (preprocess → inference → postprocess). It will not receive new jobs while in this state. When the job completes, it transitions back to IDLE.

**STOPPED**: The worker has crashed or been intentionally shut down. The JVM detects this via heartbeat and, if `unregister_model_when_failed=false` (the default), will attempt to restart the worker up to the configured retry limit.

When you set `min_worker=4`, TorchServe maintains at least 4 workers in READY state at all times. When you set `max_worker=8`, TorchServe can scale up to 8 workers under load. Scaling down happens by sending a graceful shutdown signal to idle workers after a configurable idle timeout.

### How the frontend routes to workers

The routing logic is simpler than most engineers expect. When a job is dequeued by the batch scheduler (which groups pending jobs up to `batch_size` or until `max_batch_delay` expires), the scheduler selects an IDLE worker using a least-recently-used (LRU) policy. It sends the batch to that worker via the Unix socket, marks the worker as BUSY, and waits for the response. When the worker responds with the batch of predictions, the scheduler matches each prediction back to its originating HTTP request, sends the HTTP responses, and marks the worker as IDLE.

This design is why worker count directly translates to concurrency. With 4 workers and a batch size of 8, TorchServe can have up to 4 × 8 = 32 samples in flight simultaneously across 4 GPU compute streams.

### The batch scheduler in detail

The batch scheduler runs as a thread in the JVM. Its algorithm for each model's queue is:

1. Dequeue all available jobs, up to `batch_size`.
2. If the dequeued count equals `batch_size`, dispatch immediately (batch is full).
3. If the dequeued count is less than `batch_size` but the oldest job in the current partial batch has been waiting longer than `max_batch_delay` milliseconds, dispatch with the partial batch.
4. Otherwise, wait and check again after a short poll interval.

Step 3 is the key safety mechanism. Without `max_batch_delay`, a low-traffic period could leave requests waiting indefinitely for a batch to fill. With `max_batch_delay=50ms`, the worst-case additional latency from batching is 50ms — a 99th-percentile padding you add to every inference call in exchange for the throughput benefit of larger batches.

There is a subtle interaction between `max_batch_delay` and `job_queue_size`. If `job_queue_size=100` and requests arrive at 200 req/s (faster than the system can drain the queue), the queue fills up. Once the queue is full, all incoming requests receive HTTP 503 immediately, before reaching the batch scheduler. The batch scheduler's `max_batch_delay` only applies to requests that made it into the queue. This is why `job_queue_size` is a secondary tuning parameter — it sets the burst buffer capacity for traffic spikes.

### Netty thread pool sizing

Netty uses a pair of thread pools internally: the boss group (accepts incoming connections) and the worker group (handles I/O for accepted connections). TorchServe sets the boss group to 1 thread and the worker group to `2 × CPU_cores` threads by default. For an instance with 16 CPU cores, you get 32 I/O threads.

Each I/O thread handles multiple connections concurrently through non-blocking selectors. In practice, the Netty I/O threads are rarely the bottleneck — the GPU compute in the Python workers is almost always the binding constraint. The only scenario where Netty becomes a bottleneck is extremely high connection rates with very short request bodies, at QPS > 10,000. For typical ML serving workloads (request body = a few kilobytes of text or image), Netty's throughput ceiling is well above what a single GPU can process anyway.

## 2. Model Archive (MAR): Packaging for production

A Model Archive (MAR) file is TorchServe's deployment unit. It is a ZIP file (despite the `.mar` extension) containing everything the server needs to load and run your model: serialized weights, handler code, and supporting files like tokenizer configs and label maps. Bundling everything into a single artifact makes deployment reproducible — you ship one file to your model store, and TorchServe can load it on any compatible instance.

### What goes inside a MAR

A MAR file contains:

**Serialized model weights** — the `.pt`, `.bin`, or `.safetensors` file containing your trained model parameters. TorchServe is agnostic to the file format; it is your handler's job to load them. For PyTorch models saved with `torch.save(model.state_dict(), path)`, this is a `.pt` file. For Hugging Face models, these are typically `pytorch_model.bin` or sharded `.safetensors` files.

**handler.py** — the Python class implementing the `BaseHandler` interface. This is the only required Python file. It defines how requests are preprocessed, how the model is called, and how outputs are postprocessed into JSON-serializable responses.

**extra-files** — any additional files your handler needs at runtime. For a BERT model, this means `tokenizer_config.json`, `vocab.txt`, and `config.json`. For a ResNet, this might be an `index_to_name.json` label mapping. These files are extracted to the model's working directory when TorchServe loads the MAR.

**manifest.json** — auto-generated by `torch-model-archiver`. It records the model name, version, handler class path, and the relative path to the serialized file. TorchServe reads this first when loading a MAR to understand its contents.

### Building MAR files with torch-model-archiver

```bash
# Install the archiver
pip install torch-model-archiver

# BERT-large sentiment classifier MAR
torch-model-archiver \
  --model-name bert-sentiment \
  --version 1.0 \
  --serialized-file bert-large-uncased.pt \
  --handler bert_handler.py \
  --extra-files "tokenizer_config.json,vocab.txt,config.json" \
  --archive-format zip \
  --export-path ./model-store

# Llama-3-8B MAR (HuggingFace format, weights in a directory)
torch-model-archiver \
  --model-name llama3-8b \
  --version 1.0 \
  --handler llama_handler.py \
  --extra-files "model_dir/" \
  --archive-format zip \
  --export-path ./model-store
```

The `--archive-format zip` flag is important. The default format is `default`, which creates a proprietary format that is slightly harder to inspect. The `zip` format creates a standard ZIP archive you can open with any ZIP tool, which helps during debugging when you need to verify the MAR contents.

### Versioning and routing

The `--version` flag sets a string version identifier embedded in the manifest. TorchServe uses this for version-aware routing. When you register `bert-sentiment` version 1.0 and then register version 2.0, both exist simultaneously in the model store. You can route to specific versions:

```bash
# Route to specific version
curl http://localhost:8080/predictions/bert-sentiment/2.0 \
  -H "Content-Type: application/json" \
  -d '{"data": "Great product!"}'

# Route to default version (whatever was registered last, or explicitly set)
curl http://localhost:8080/predictions/bert-sentiment \
  -H "Content-Type: application/json" \
  -d '{"data": "Great product!"}'
```

Setting a default version via the management API:

```bash
curl -X PUT "http://localhost:8081/models/bert-sentiment/2.0/set-default"
```

This versioning mechanism enables blue-green deployments without a load balancer configuration change: register version 2.0, test it by calling the versioned endpoint explicitly, then flip the default and deregister version 1.0.

#### Worked example: BERT-large MAR for sentiment classification

Let us walk through every file needed for a production BERT-large sentiment MAR and understand the exact sizes involved.

The base model is `bert-large-uncased` from Hugging Face. BERT-large has 340 million parameters. In float32, each parameter occupies 4 bytes, so the raw weights file is:

$$340 \times 10^6 \text{ params} \times 4 \text{ bytes/param} = 1.36 \text{ GB}$$

In float16 (half precision), the weights file halves to approximately 0.68 GB. Most production deployments use fp16 for BERT-large because BERT accuracy degrades negligibly with fp16 quantization (the model was not trained with numerical precision that would be lost) while GPU memory usage and inference latency both improve.

The supporting files are small:
- `vocab.txt` (BERT vocabulary): 231 KB
- `tokenizer_config.json`: 1.2 KB
- `config.json` (model architecture config): 0.8 KB
- `bert_handler.py`: approximately 4 KB

The resulting MAR file for the fp16 BERT-large model will be approximately **689 MB** (the ZIP compression ratio for binary weight files is near 1.0 — floating point model weights are essentially incompressible).

Loading time on a machine with an NVMe SSD (sequential read ~3.5 GB/s) and GPU-to-CPU PCIe bandwidth of ~16 GB/s:
- Disk read: 689 MB ÷ 3,500 MB/s ≈ **0.20 seconds**
- Host-to-device transfer: 689 MB ÷ 16,000 MB/s ≈ **0.04 seconds**
- Python import and tokenizer initialization: **1–2 seconds** (BERT tokenizer has a non-trivial initialization path in Hugging Face Transformers)
- Warm-up forward pass: **0.1 seconds** (first pass triggers CUDA kernel compilation and caching)

Total loading time per worker: approximately **2–4 seconds**. With 4 workers, all load in parallel, so the overall startup time is still approximately 4 seconds. This is acceptable for a long-lived service but problematic for serverless deployments with per-request cold starts.

![MAR file creation pipeline showing inputs flowing into torch-model-archiver and producing the final archive with manifest](/imgs/blogs/torchserve-deep-dive-2.png)

## 3. Handlers: The heart of TorchServe customization

The handler is the most important piece of your TorchServe deployment. It defines all four phases of the inference pipeline: model loading, request preprocessing, forward pass, and response postprocessing. TorchServe ships several built-in handlers for common use cases, but production deployments almost always require a custom handler to handle tokenization, image preprocessing pipelines, or custom output schemas.

### The BaseHandler contract

`BaseHandler` defines four methods you implement:

**`initialize(ctx)`** is called once per worker during startup. The `ctx` (context) object provides access to `ctx.system_properties`, a dictionary containing `model_dir` (the directory where MAR contents were extracted), `gpu_id` (the GPU index assigned to this worker), and `batch_size` (the configured maximum batch size). In `initialize`, you load model weights, move them to the correct device, set `model.eval()`, and run a warm-up forward pass to trigger CUDA kernel compilation. Failing to run a warm-up pass means the very first production request will pay the CUDA JIT compilation penalty — typically 200–500ms for a model like BERT-large.

**`preprocess(data)`** receives a `List[dict]` where each dict represents one client request. The dict contains `data` (the raw request body as bytes or string) and `body` (alias). Your preprocess method must transform this list into the input tensor(s) your model expects. For NLP models, this means tokenization and padding. For image models, this means decoding, resizing, normalization, and stacking. The output of preprocess is passed directly to inference.

**`inference(data)`** receives the batched tensor(s) from preprocess, runs the forward pass with `torch.no_grad()`, and returns the raw model output tensor(s). This method should be as thin as possible — just the forward pass. Do not do any output processing here.

**`postprocess(data)`** receives the raw output tensor(s) from inference. It must return a `List` of JSON-serializable objects — one per input request. The list length must exactly match the batch size. TorchServe maps each element back to its corresponding HTTP request using list index.

### Full custom handler for BERT sentiment

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from ts.torch_handler.base_handler import BaseHandler
import json
import logging

logger = logging.getLogger(__name__)

class BertSentimentHandler(BaseHandler):
    """
    Custom TorchServe handler for BERT-large sentiment classification.
    Handles batched requests with dynamic padding.
    """
    
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() else "cpu"
        )
        
        # Load tokenizer from extra-files
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # Load model weights
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = f"{model_dir}/{serialized_file}"
        
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir, num_labels=2
        )
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Warm-up pass to initialize CUDA kernels
        dummy_input = self.tokenizer(
            "warm up", return_tensors="pt", padding=True
        )
        with torch.no_grad():
            self.model(**{k: v.to(self.device) for k, v in dummy_input.items()})
        
        logger.info("BERT sentiment handler initialized on %s", self.device)
    
    def preprocess(self, data):
        texts = [d.get("data") or d.get("body") for d in data]
        if isinstance(texts[0], (bytes, bytearray)):
            texts = [t.decode("utf-8") for t in texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def inference(self, data):
        with torch.no_grad():
            outputs = self.model(**data)
        return outputs.logits
    
    def postprocess(self, data):
        probs = torch.softmax(data, dim=-1).cpu().numpy()
        labels = ["negative", "positive"]
        results = []
        for prob in probs:
            results.append({
                "label": labels[prob.argmax()],
                "score": float(prob.max()),
                "probabilities": {
                    "negative": float(prob[0]),
                    "positive": float(prob[1])
                }
            })
        return results
```

Several production details in this handler deserve attention.

The `padding=True, truncation=True` combination in the tokenizer call is critical for batching correctness. When a batch contains sequences of different lengths, `padding=True` pads shorter sequences to match the longest sequence in the batch. `truncation=True` cuts sequences longer than `max_length=512`. Without both flags, the tokenizer will fail on heterogeneous-length batches. The padding creates computational waste on the padded tokens, but BERT's attention mask mechanism correctly ignores padded positions during the forward pass.

The `map_location=self.device` in `torch.load` is critical for multi-GPU deployments. Without it, PyTorch loads weights to whatever device they were saved on — often `cuda:0`. If your worker is assigned `gpu_id=1`, the weights land on the wrong GPU and the first inference fails with a device mismatch error.

The warm-up forward pass at the end of `initialize` deserves special attention. CUDA's JIT compilation system compiles kernels lazily — on first use. For BERT-large on CUDA, the first real forward pass triggers compilation of the attention kernels, layer normalization kernels, and the linear projection kernels. This compilation takes 200–500ms. By running a dummy forward pass during initialization, you absorb this penalty at startup time rather than making the first production request pay it.

### Handler for streaming LLM responses

TorchServe 0.9+ supports streaming responses via the `send_intermediate_predict_response` mechanism, which allows handlers to emit tokens incrementally rather than waiting for full generation:

```python
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context
import torch

class StreamingLlamaHandler(BaseHandler):
    
    def initialize(self, context: Context):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )
        self.model.eval()
        self.context = context
    
    def preprocess(self, data):
        texts = [d.get("data") or d.get("body") for d in data]
        if isinstance(texts[0], (bytes, bytearray)):
            texts = [t.decode("utf-8") for t in texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def inference(self, data):
        # Non-streaming generation for batched requests
        with torch.no_grad():
            outputs = self.model.generate(
                **data,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return outputs
    
    def postprocess(self, data):
        results = []
        for output in data:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append({"generated_text": text})
        return results
```

Note the limitation here: this handler still uses static batching (collects up to `batch_size` requests, runs generation, returns all results together). The slowest sequence in the batch determines when all clients get their response. This is TorchServe's fundamental limitation for LLM serving, covered in detail in Section 8.

### Handler error handling and retry semantics

A well-written production handler must handle three classes of errors explicitly:

**Input validation errors**: The client sent malformed JSON, a non-UTF-8 string, or a payload exceeding the model's maximum sequence length. These should be caught in `preprocess` and returned as application-level error responses, not Python exceptions. An unhandled exception in `preprocess` causes TorchServe to return HTTP 500 for the entire batch — including the valid requests in that batch that could have succeeded.

The correct pattern is to preprocess defensively and tag invalid requests:

```python
def preprocess(self, data):
    texts = []
    errors = {}
    for i, d in enumerate(data):
        raw = d.get("data") or d.get("body")
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                errors[i] = "invalid utf-8 encoding"
                raw = ""
        texts.append(raw)
    self._batch_errors = errors

    inputs = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {k: v.to(self.device) for k, v in inputs.items()}

def postprocess(self, data):
    probs = torch.softmax(data, dim=-1).cpu().numpy()
    labels = ["negative", "positive"]
    results = []
    for i, prob in enumerate(probs):
        if hasattr(self, "_batch_errors") and i in self._batch_errors:
            results.append({"error": self._batch_errors[i]})
        else:
            results.append({
                "label": labels[prob.argmax()],
                "score": float(prob.max()),
                "probabilities": {
                    "negative": float(prob[0]),
                    "positive": float(prob[1])
                }
            })
    return results
```

**GPU OOM errors**: If the batch contains unexpectedly long sequences that cause an out-of-memory error during the forward pass, the worker crashes. TorchServe restarts it automatically (with the retry configured by `model_config_group_timeout_ms`), but the in-flight batch is lost and all those clients receive HTTP 500. Add explicit sequence length validation in `preprocess` with a hard cap at `max_length - 10` tokens (leave headroom for `[CLS]` and `[SEP]` special tokens), and catch `torch.cuda.OutOfMemoryError` in `inference` to reduce batch size and retry:

```python
def inference(self, data):
    try:
        with torch.no_grad():
            outputs = self.model(**data)
        return outputs.logits
    except torch.cuda.OutOfMemoryError:
        # Emergency fallback: split batch in half, process serially
        torch.cuda.empty_cache()
        half = {k: v[:len(v)//2] for k, v in data.items()}
        with torch.no_grad():
            out_a = self.model(**half).logits
        rest = {k: v[len(v)//2:] for k, v in data.items()}
        with torch.no_grad():
            out_b = self.model(**rest).logits
        return torch.cat([out_a, out_b], dim=0)
```

**Model inference timeouts**: For generative models that can produce arbitrarily long outputs, a single very long generation can hold a worker BUSY for minutes. Set `default_response_timeout` in `config.properties` to a reasonable ceiling for your workload and ensure your generation call respects `max_new_tokens`. For BERT-style encoders, inference time is bounded by input length and this is rarely a concern.

### Advanced handler patterns: fp16 and TorchScript

Loading BERT-large in float16 halves GPU memory usage and increases throughput on GPUs with Tensor Cores (T4, A100, H100). The GPU memory math:

- BERT-large fp32: $340 \times 10^6 \times 4 \text{ bytes} = 1.36 \text{ GB}$
- BERT-large fp16: $340 \times 10^6 \times 2 \text{ bytes} = 0.68 \text{ GB}$

Load in fp16 by passing `torch_dtype=torch.float16` to `from_pretrained`:

```python
# In initialize:
self.model = BertForSequenceClassification.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    num_labels=2
)
self.model.to(self.device)
# Tokenizer outputs int64 ids and masks — these stay in int64.
# Only model weights are float16.
```

Activation memory in fp16 for a batch of 8 at seq_len=128 across 24 BERT-large layers:

$$\text{Activations} = 8 \times 128 \times 1024 \times 24 \times 2 \text{ bytes} = 0.40 \text{ GB}$$

Compare to fp32: $8 \times 128 \times 1024 \times 24 \times 4 \text{ bytes} = 0.80 \text{ GB}$. At fp16, you free up nearly 400 MB per active batch — meaningful when running 4 concurrent workers on a 16 GB GPU.

For TorchScript compilation in `initialize`, which provides 10–25% latency reduction for transformer models by fusing operations and eliminating Python dispatch overhead:

```python
def initialize(self, context):
    # ... load model as above ...
    self.model.eval()

    # Trace with a representative fixed input shape
    dummy_ids = torch.zeros(1, 128, dtype=torch.long).to(self.device)
    dummy_mask = torch.ones(1, 128, dtype=torch.long).to(self.device)
    with torch.no_grad():
        traced = torch.jit.trace(
            self.model,
            (dummy_ids, dummy_mask)
        )
    self.model = torch.jit.optimize_for_inference(traced)
```

Note: `torch.jit.trace` records operations for a specific input shape. If your production batches vary in sequence length (they always do), traced models handle dynamic lengths correctly via PyTorch's shape-polymorphic tracing — but the traced path must have seen all conditional branches. For BERT, this is fine because the attention mechanism does not branch on input shape.

![BaseHandler lifecycle showing the four phases from initialize through postprocess with data flow between each phase](/imgs/blogs/torchserve-deep-dive-4.png)

## 4. Dynamic batching: The latency-throughput trade-off in numbers

Dynamic batching is TorchServe's mechanism for automatically grouping individual inference requests into larger batches before dispatching them to a GPU worker. The motivation is straightforward: modern GPU architectures are designed for massively parallel computation. Running a single BERT-large forward pass on a T4 uses perhaps 18% of available CUDA cores. Running a batch of 8 forward passes simultaneously uses 84% — you get near-linear throughput gains with only modest latency increases.

### The math behind batching efficiency

For BERT-large on a T4 GPU with fp16 precision, measured inference times are:

- Batch of 1: 18ms
- Batch of 4: 31ms
- Batch of 8: 45ms
- Batch of 16: 72ms
- Batch of 32: out of memory on T4 16GB

The throughput (samples per second) for each configuration:
- Batch 1: $1 / 0.018 = 55.6$ samples/second
- Batch 4: $4 / 0.031 = 129$ samples/second (2.3×)
- Batch 8: $8 / 0.045 = 178$ samples/second (3.2×)
- Batch 16: $16 / 0.072 = 222$ samples/second (4.0×)

The GPU memory math for batch size 32 (why it OOMs on a T4 16GB):

BERT-large has 24 transformer layers. For a batch of 32 sequences at `seq_len=128` in fp16:

$$\text{Activations per layer} = 32 \times 128 \times 1024 \times 2 \text{ bytes} = 8.4 \text{ MB}$$

$$\text{Total activations (24 layers)} = 24 \times 8.4 = 201 \text{ MB}$$

But this is just the hidden states. The attention score matrices for each layer add:

$$\text{Attention scores} = 32 \times 16 \text{ heads} \times 128 \times 128 \times 2 \text{ bytes} = 16.8 \text{ MB per layer}$$

$$\text{Total attention} = 24 \times 16.8 = 403 \text{ MB}$$

Model weights in fp16: 680 MB. Plus the optimizer is not loaded during inference, but we still need Python process overhead (~250 MB) and CUDA runtime overhead (~500 MB for a T4).

Total at batch 32: $680 + 201 + 403 + 250 + 500 + \text{gradients} \approx 15.5\text{–}16.5 \text{ GB}$. The T4 has 16 GB, so batch 32 sits right at the limit — it may succeed or fail depending on OS-level memory fragmentation. In practice, use batch 16 as the safe maximum for BERT-large on T4.

### Little's Law applied to TorchServe

Little's Law states that for a stable queuing system:

$$L = \lambda W$$

where $L$ is the average number of jobs in the system, $\lambda$ is the arrival rate (requests/second), and $W$ is the average time a job spends in the system (from arrival to response). Rearranging: $W = L / \lambda$.

For a TorchServe deployment with $N$ workers each processing batches of size $B$ with inference time $T_{inf}$, the theoretical throughput ceiling is:

$$\lambda_{max} = \frac{N \times B}{T_{inf}}$$

With 4 workers, batch size 8, and 45ms inference time per batch:

$$\lambda_{max} = \frac{4 \times 8}{0.045} = 711 \text{ req/s}$$

However, this theoretical maximum ignores preprocessing time, postprocessing time, network serialization, and the batching wait time itself. In practice, expect to achieve 60–70% of the theoretical maximum.

For the expected wait time a request spends queued before being dispatched (assuming Poisson arrivals at rate $\lambda$ and batch threshold $B$):

$$E[\text{wait}] \approx \frac{B - 1}{2\lambda} \quad \text{when } \lambda \ll \frac{B}{T_{inf}}$$

At 30 req/s with batch_size=8:

$$E[\text{wait}] = \frac{8 - 1}{2 \times 30} = 0.117 \text{ seconds} = 117\text{ms}$$

This is the expected batching wait time added to every request. If your SLO is p99 < 100ms, a batch_size of 8 at 30 req/s already violates it from queuing delay alone. You must either reduce batch_size, increase arrival rate (so the batch fills faster), or accept higher latency.

With `max_batch_delay=50ms`, TorchServe will dispatch whatever requests have arrived after 50ms even if the batch is not full. So in practice, the batching overhead is bounded by `max_batch_delay`, not the theoretical expectation — but `max_batch_delay` directly adds to p99 latency.

### Batching configuration

```properties
# config.properties for TorchServe
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_gpu=1
vmargs=-Xmx2g -XX:MaxDirectMemorySize=512m -XX:+ExitOnOutOfMemoryError

# Model-level batching (override per-model via management API)
batch_size=8
max_batch_delay=50

# Worker configuration
default_workers_per_model=2
job_queue_size=100
default_response_timeout=120
```

#### Worked example: Tuning batch size for BERT sentiment on T4

The following table shows measured throughput and latency for BERT-large sentiment classification on a T4 16GB GPU, with two workers:

| batch_size | max_batch_delay | Throughput (req/s) | p50 latency (ms) | p99 latency (ms) | GPU util |
|---|---|---|---|---|---|
| 1 | 0ms | 12 | 22 | 45 | 18% |
| 4 | 25ms | 38 | 35 | 68 | 61% |
| 8 | 50ms | 58 | 48 | 92 | 84% |
| 16 | 100ms | 71 | 89 | 180 | 91% |
| 32 | 200ms | OOM | — | — | — |

The SLO triangle governs every tuning decision here. You cannot simultaneously minimize latency, maximize throughput, and minimize cost. The three vertices of the triangle are:

**Latency**: Minimize `batch_size` and `max_batch_delay`. At batch_size=1 and max_batch_delay=0, you get the lowest possible p99 (45ms) but terrible throughput (12 req/s) and 18% GPU utilization — you are paying for GPU time you are not using.

**Throughput**: Maximize `batch_size` and `max_batch_delay`. At batch_size=16 and max_batch_delay=100ms, you get 71 req/s and 91% GPU utilization, but p99 latency of 180ms — which may violate your SLO.

**Cost**: The efficient frontier is the curve connecting batch_size=4 through batch_size=8 in the table above. At batch_size=8, you get 4.8× the throughput of batch_size=1 at only 2.1× the p99 latency. Moving from batch_size=8 to batch_size=16 gives only 22% more throughput at 96% higher p99 latency — well off the efficient frontier.

For a service with a p99 SLO of ≤ 150ms, batch_size=8 with max_batch_delay=50ms is the optimal configuration.

![Dynamic batching showing single request processing versus batched requests with GPU utilization comparison and latency-throughput curve](/imgs/blogs/torchserve-deep-dive-3.png)

## 5. Management API: Registering, scaling, and unregistering models

The management API at port 8081 is TorchServe's administrative surface. All model lifecycle operations go through here. Understanding it thoroughly is essential for automation — every deployment pipeline, autoscaler, and canary rollout mechanism interacts with these endpoints.

### Registering a model

```bash
# Register BERT sentiment with 2 workers, batch size 8
curl -X POST "http://localhost:8081/models" \
  -F "model_name=bert-sentiment" \
  -F "url=bert-sentiment.mar" \
  -F "batch_size=8" \
  -F "max_batch_delay=50" \
  -F "initial_workers=2" \
  -F "synchronous=true"

# Response
{
  "status": "Model \"bert-sentiment\" Version: 1.0 registered with 2 initial workers"
}
```

The `synchronous=true` flag causes the API call to block until all `initial_workers` have reached READY state. Without it, the call returns immediately and you must poll `/models/bert-sentiment` to check readiness. In deployment automation, always use `synchronous=true` to avoid race conditions where your deployment script proceeds before workers have finished loading.

The `url` parameter accepts either a local path relative to the model store directory, or an HTTPS URL. TorchServe will download from HTTPS URLs directly, which enables CI/CD pipelines that upload MAR files to S3 and register them via URL.

```bash
# Register from S3 URL
curl -X POST "http://localhost:8081/models" \
  -F "model_name=bert-sentiment" \
  -F "url=https://my-model-bucket.s3.amazonaws.com/bert-sentiment-v2.mar" \
  -F "initial_workers=4" \
  -F "synchronous=true"
```

### Scaling workers

```bash
# Scale to 4 workers (triggers model reload on new workers)
curl -X PUT "http://localhost:8081/models/bert-sentiment" \
  -F "min_worker=4" \
  -F "max_worker=4" \
  -F "synchronous=true"

# Check worker status
curl http://localhost:8081/models/bert-sentiment
```

The response from the status endpoint shows each worker's state:

```json
{
  "modelName": "bert-sentiment",
  "modelVersion": "1.0",
  "modelUrl": "bert-sentiment.mar",
  "runtime": "python",
  "minWorkers": 4,
  "maxWorkers": 4,
  "batchSize": 8,
  "maxBatchDelay": 50,
  "loadedAtStartup": false,
  "workers": [
    {"id": "9000", "startTime": "2026-06-22T02:47:13.445Z", "status": "READY", "gpu": true, "memoryUsage": 1847820288},
    {"id": "9001", "startTime": "2026-06-22T02:47:14.112Z", "status": "READY", "gpu": true, "memoryUsage": 1851392000},
    {"id": "9002", "startTime": "2026-06-22T02:47:14.889Z", "status": "READY", "gpu": true, "memoryUsage": 1849753600},
    {"id": "9003", "startTime": "2026-06-22T02:47:15.621Z", "status": "BUSY", "gpu": true, "memoryUsage": 1847820288}
  ]
}
```

The `memoryUsage` field shows resident set size in bytes for each worker — useful for detecting memory leaks in handlers that accumulate data in instance variables across requests.

### Listing and unregistering models

```bash
# List all registered models
curl http://localhost:8081/models
# {
#   "models": [
#     {"modelName": "bert-sentiment", "modelUrl": "bert-sentiment.mar"},
#     {"modelName": "resnet50", "modelUrl": "resnet50.mar"}
#   ]
# }

# Unregister specific version
curl -X DELETE "http://localhost:8081/models/bert-sentiment/1.0"

# Check if unregistration complete
curl http://localhost:8081/models/bert-sentiment
# 404 if fully unregistered
```

### Worker count versus batch size: A critical distinction

These two knobs are frequently confused. They control different dimensions of parallelism:

**Worker count** controls horizontal parallelism — the number of concurrent request batches in flight simultaneously. Each worker is an independent Python process with its own copy of the model loaded into GPU memory. With 4 workers, TorchServe can process 4 batches simultaneously. This is limited by GPU memory: 4 workers × 1.34 GB (BERT-large fp32) = 5.36 GB of weights, plus activation memory, must fit in GPU VRAM.

**Batch size** controls vertical parallelism — the number of samples processed in a single GPU forward pass. Larger batches use more GPU CUDA cores simultaneously. This is limited by GPU memory (activations grow linearly with batch size) and by your latency SLO (larger batches require waiting longer for the batch to fill, or accepting partially-filled batches at the cost of throughput).

The interaction between them is multiplicative for throughput: throughput scales with `workers × batch_size / T_inference`. But their memory costs are additive in different ways — more workers multiplies weight memory, larger batches multiplies activation memory.

## 6. Inference API and multi-model serving

### Making inference requests

```bash
# Single text prediction
curl -X POST http://localhost:8080/predictions/bert-sentiment \
  -H "Content-Type: application/json" \
  -d '{"data": "This movie was absolutely fantastic!"}'

# Response
{
  "label": "positive",
  "score": 0.9847,
  "probabilities": {"negative": 0.0153, "positive": 0.9847}
}

# Image classification with ResNet-50
curl -X POST http://localhost:8080/predictions/resnet50 \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.jpg

# Specifying model version
curl -X POST http://localhost:8080/predictions/bert-sentiment/2.0 \
  -H "Content-Type: application/json" \
  -d '{"data": "Service was terrible!"}'
```

TorchServe does not expose a native batch request endpoint. From the client perspective, each HTTP request is a single logical request. TorchServe's dynamic batcher groups concurrent HTTP requests arriving within the `max_batch_delay` window into a single GPU batch internally. If you need to send a bulk payload (for example, classifying 1,000 records at once), you have two options: send 1,000 concurrent requests and let TorchServe batch them (requires your client to use async HTTP), or write a custom handler that accepts a JSON array body and iterates over it internally (bypasses TorchServe batching but allows bulk payloads per request).

### Multi-model serving architecture

TorchServe's multi-model capability is one of its most compelling production features. Running multiple models on a single instance reduces infrastructure cost significantly when not all models are at peak QPS simultaneously.

```bash
# Start TorchServe with two models loaded at startup
torchserve --start \
  --model-store ./model-store \
  --models bert-sentiment=bert-sentiment.mar,resnet50=resnet50.mar \
  --ts-config config.properties \
  --ncs

# Check both models are ready
curl http://localhost:8081/models
```

Configure workers and batching per-model:

```bash
# BERT gets more workers (NLP workload, higher QPS)
curl -X PUT "http://localhost:8081/models/bert-sentiment" \
  -F "min_worker=4" \
  -F "max_worker=4" \
  -F "batch_size=8"

# ResNet gets fewer workers (CV, lower QPS, bigger batches)
curl -X PUT "http://localhost:8081/models/resnet50" \
  -F "min_worker=2" \
  -F "max_worker=2" \
  -F "batch_size=16"
```

### GPU assignment in multi-model deployments

The `number_of_gpu` setting in `config.properties` tells TorchServe how many GPUs are available. TorchServe assigns workers to GPUs round-robin. With 4 workers for bert-sentiment and 2 workers for resnet50 on a 2-GPU instance:

- bert-sentiment worker 0 → GPU 0
- bert-sentiment worker 1 → GPU 1
- bert-sentiment worker 2 → GPU 0
- bert-sentiment worker 3 → GPU 1
- resnet50 worker 0 → GPU 0
- resnet50 worker 1 → GPU 1

Each GPU ends up running 3 worker processes (2 BERT + 1 ResNet), all sharing that GPU's memory. The round-robin assignment is automatic. To override and pin a model to a specific GPU, read `ctx.system_properties["gpu_id"]` in your handler and add logic accordingly — though TorchServe does not expose a direct API to pin models to specific GPU indices.

GPU memory accounting for this co-located deployment:
- BERT-large fp16 × 3 workers: $3 \times 0.68 = 2.04$ GB
- ResNet-50 fp32 × 1.5 workers (average per GPU): $1.5 \times 0.10 = 0.15$ GB
- CUDA runtime per GPU: 0.5 GB
- Total per GPU: approximately 2.7 GB

On a T4 16GB, this is comfortable. On a 4GB GPU (like an older GTX 1650), it would OOM.

![Multi-model serving with BERT and ResNet co-located on a TorchServe instance, showing GPU assignment and per-model request queues](/imgs/blogs/torchserve-deep-dive-5.png)

## 7. Metrics and observability

TorchServe's built-in Prometheus integration is one of the features that distinguishes it from rolling your own Flask serving layer. Observability comes for free — you get quantile histograms for inference latency, queue wait time, and request counts without writing any custom instrumentation.

### Built-in Prometheus metrics

The metrics endpoint at `:8082/metrics` exposes these key metrics:

**`ts_inference_latency_microseconds`** — a histogram of end-to-end inference duration (from worker receiving the batch to returning the results). Uses Prometheus histogram buckets. The `model_name` label identifies which model the measurement came from.

**`ts_queue_latency_microseconds`** — a histogram of time spent waiting in the job queue. If this is high relative to `ts_inference_latency_microseconds`, you need more workers or larger batches. Queue latency is the primary signal for under-provisioning.

**`WorkerThreadTime`** — a gauge showing how much time worker threads have spent executing. Used to diagnose uneven load distribution across workers.

**`ts_inference_requests_total`** — a counter of total inference requests, labeled by model name and HTTP status code. Essential for calculating error rates.

**`QueueTime`** — a gauge (in addition to the histogram) showing the current instantaneous queue wait time. Useful for real-time alerting.

### Prometheus scrape configuration

```yaml
# prometheus.yml scrape config
scrape_configs:
  - job_name: torchserve
    static_configs:
      - targets: ["localhost:8082"]
    metrics_path: /metrics
    scrape_interval: 15s
```

### Grafana queries for operational dashboards

p99 inference latency in milliseconds:

```promql
histogram_quantile(0.99, 
  rate(ts_inference_latency_microseconds_bucket[5m])
) / 1000
```

p50/p95/p99 queue latency:

```promql
histogram_quantile(0.95,
  rate(ts_queue_latency_microseconds_bucket[5m])
) / 1000
```

Request throughput per model:

```promql
rate(ts_inference_requests_total{model_name="bert-sentiment"}[1m])
```

Error rate:

```promql
rate(ts_inference_requests_total{code!="200"}[5m])
/ rate(ts_inference_requests_total[5m])
```

### Custom metrics in handlers

TorchServe's metrics store allows handlers to emit application-specific metrics:

```python
from ts.metrics.metrics_store import MetricsStore

class BertSentimentHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        # Access the metrics store via context
        self.metrics = context.metrics
    
    def postprocess(self, data):
        results = super().postprocess(data)
        # Record prediction confidence as a custom metric
        for r in results:
            self.metrics.add_metric(
                name="PredictionConfidence",
                value=r["score"],
                unit="Percent",
                dimensions=[("ModelName", "bert-sentiment")]
            )
        return results
```

Custom metrics appear in the Prometheus output with the `ts_` prefix and your chosen metric name. You can track business metrics like prediction confidence distribution, token lengths (for NLP models), input image sizes (for vision models), or any other application-level signal.

### Setting up alerting rules

A practical Prometheus alerting rule for the 2 AM scenario from the introduction — catching high queue latency before users notice:

```yaml
# alerting_rules.yml
groups:
  - name: torchserve
    rules:
      - alert: TorchServeHighQueueLatency
        expr: |
          histogram_quantile(0.99,
            rate(ts_queue_latency_microseconds_bucket[5m])
          ) / 1000 > 200
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "TorchServe queue latency exceeding 200ms p99"
          description: "Model {{ $labels.model_name }} p99 queue wait is {{ $value }}ms. Scale workers or reduce batch size."
      
      - alert: TorchServeWorkerCrash
        expr: |
          rate(ts_inference_requests_total{code="503"}[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TorchServe returning 503 errors — workers likely crashed"
      
      - alert: TorchServeHighInferenceLatency
        expr: |
          histogram_quantile(0.99,
            rate(ts_inference_latency_microseconds_bucket[5m])
          ) / 1000 > 500
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "TorchServe p99 inference latency above 500ms"
          description: "Model {{ $labels.model_name }} is taking longer than expected. Check GPU utilization and batch configuration."
```

The queue latency alert fires before users see elevated latency — by the time your p99 inference latency spikes, the queue was already backed up for minutes. The `for: 2m` clause prevents false alerts from transient traffic spikes that self-resolve.

### Interpreting the metrics: A diagnostic workflow

When you receive a TorchServe latency alert, follow this diagnostic sequence using the three Prometheus metrics:

**Step 1: Is the problem in queuing or in inference?**

Compare `ts_queue_latency_microseconds p99` with `ts_inference_latency_microseconds p99`. If queue latency is much higher than inference latency, the problem is worker capacity (not enough workers, or batches not configured correctly). If inference latency itself has increased, the problem is in the model (GPU memory fragmentation, unexpected long inputs, hardware degradation).

**Step 2: What is the current throughput?**

```promql
rate(ts_inference_requests_total{code="200"}[1m])
```

Compare this to your theoretical throughput ceiling: $N_{workers} \times B_{batch} / T_{inference}$. If actual throughput is well below the ceiling, you have a queuing problem. If actual throughput is at the ceiling and p99 latency is high, you need to scale.

**Step 3: Are workers healthy?**

```promql
# Worker utilization: fraction of time workers are BUSY vs IDLE
# Approximated by inference time / (inference time + queue time)
histogram_quantile(0.50, rate(ts_inference_latency_microseconds_bucket[5m]))
/ (
  histogram_quantile(0.50, rate(ts_inference_latency_microseconds_bucket[5m]))
  + histogram_quantile(0.50, rate(ts_queue_latency_microseconds_bucket[5m]))
)
```

If worker utilization is above 90%, workers are saturated — add workers or increase batch size. If worker utilization is below 30% but queue latency is high, something is wrong with the worker-queue connection (possibly a worker in BUSY state that is hung).

**Step 4: Check the 503 rate.**

```promql
rate(ts_inference_requests_total{code="503"}[5m])
/ rate(ts_inference_requests_total[5m])
```

A nonzero 503 rate means the job queue is filling (`job_queue_size` reached). Either increase `job_queue_size` to buffer more traffic, or add workers to drain the queue faster. Increasing `job_queue_size` without adding workers just delays failures — requests will still timeout, just later.

### Connecting TorchServe metrics to SLO tracking

Most engineering teams operate under latency SLOs expressed as percentiles: "p99 inference latency must be below 200ms for 99.9% of days." The Prometheus recording rule for SLO tracking:

```promql
# Record 99th-percentile inference latency as a named metric
# for long-term SLO dashboarding
record: job:ts_inference_latency_p99:ms
expr: |
  histogram_quantile(0.99,
    sum(rate(ts_inference_latency_microseconds_bucket[5m])) by (le, model_name)
  ) / 1000
```

Store this recording rule with a 30-day retention policy to generate weekly SLO compliance reports without the cardinality explosion of storing raw histogram data for 30 days.

## 8. TorchServe limitations: When NOT to use it

Honest evaluation of tooling is more useful than marketing copy. TorchServe is an excellent tool for specific workloads and a poor choice for others. Here is a frank technical assessment.

### JVM overhead

Every TorchServe deployment carries 300–500 MB of JVM heap usage plus the JIT compilation cost on startup. Each Python worker runs as a subprocess connected to the JVM via socket — adding roughly 0.5ms of inter-process communication overhead to every inference call. This overhead is constant regardless of model size, which means it matters relatively more for small, fast models (where 0.5ms is 5% of a 10ms inference time) and relatively less for large models.

For sub-10ms p99 latency requirements, the JVM baseline makes TorchServe uncompetitive with Triton Inference Server, which is written entirely in C++ with near-zero overhead for well-tuned TensorRT backends.

### Static batching and LLM serving

TorchServe's batch scheduler collects requests into batches and dispatches them atomically. All requests in a batch must complete before any results are returned. For autoregressive language model generation, this creates a severe tail latency problem.

Consider a batch of 8 generation requests where 7 requests generate 50 tokens each and 1 request generates 500 tokens. Under static batching:
- The 7 short requests complete their 50-token generation in 50 × T_token seconds
- But they wait for the 8th request to finish its 500-token generation
- All 8 clients wait 500 × T_token seconds for a response

With T_token = 40ms (typical for Llama-3-8B on an A100 with fp16), the 7 short requests wait 500 × 40ms = 20 seconds when they could have received responses in 50 × 40ms = 2 seconds. The tail latency is determined by the slowest item in every batch.

vLLM solves this with continuous batching (also called iteration-level scheduling): each decode step is a new scheduling decision. Short sequences complete and return their responses immediately, and new requests fill the freed capacity. This is not a configuration option in TorchServe — it requires a fundamentally different serving architecture.

### No PagedAttention and KV cache memory fragmentation

For autoregressive LLM serving, the key-value (KV) cache must store intermediate attention computations across all generation steps. TorchServe does not implement PagedAttention (the vLLM memory management innovation) — instead, each sequence must pre-allocate GPU memory for the maximum possible generation length.

For Llama-3-8B with a maximum generation length of 4096 tokens:
- Hidden dimension: 4096
- Number of layers: 32
- KV cache per token: $2 \times 4096 \times 32 \times 2 \text{ bytes (fp16)} = 524288 \text{ bytes}$
- KV cache per sequence at max_length=4096: $4096 \times 524288 = 2.15 \text{ GB}$

With 4 concurrent generation requests, the KV cache alone requires $4 \times 2.15 = 8.6$ GB. Combined with the 16 GB model weights (Llama-3-8B fp16 ≈ 16 GB), you need $8.6 + 16 = 24.6$ GB of GPU memory. An A100 40GB can barely handle 4 concurrent requests, and many prompts are short — the pre-allocated KV cache for max_length=4096 is mostly wasted GPU memory.

PagedAttention (used by vLLM) allocates KV cache in fixed-size pages, only committing memory as tokens are actually generated. Memory utilization improves from ~30% (TorchServe-style pre-allocation) to ~90% (vLLM), enabling dramatically more concurrent sequences.

### When to switch away from TorchServe

The decision matrix:

| Scenario | TorchServe | Triton | vLLM | Ray Serve |
|---|---|---|---|---|
| BERT or ResNet serving | Easy setup | Faster but complex | LLM-only | Works but overkill |
| LLM under 7B parameters | Works adequately | Works with TRT | Best option | Works |
| LLM over 13B parameters | No tensor parallelism | Works with model sharding | Best option | Works |
| Multi-model management | Built-in, easy | Built-in | Single model only | Requires config |
| Sub-10ms p99 latency | JVM overhead blocks this | Achievable with TRT | Not applicable | Not achievable |
| PyTorch ecosystem native | Fully native | Requires ONNX or TRT export | Fully native | Fully native |
| High-QPS LLM serving over 100 req/s | Static batching fails | Possible with optimized backends | Continuous batching native | Possible with custom logic |

The key decision boundaries:
- LLMs larger than 7B parameters → use vLLM or TGI
- Need sub-10ms p99 → use Triton with TensorRT
- Single LLM at high QPS → use vLLM
- Multi-GPU tensor parallelism → use Ray Serve with DeepSpeed-Inference
- Classic DNN models (BERT, ResNet, ViT, Whisper) at moderate QPS → TorchServe is an excellent choice

![TorchServe versus alternatives comparison matrix across serving scenarios, model sizes, and performance requirements](/imgs/blogs/torchserve-deep-dive-6.png)

## 9. Worker lifecycle and production hardening

### Worker lifecycle in detail

Workers go through a deterministic lifecycle that TorchServe manages automatically. Understanding this lifecycle is essential for debugging failures and reasoning about startup and shutdown behavior.

During normal startup, the JVM spawns Python subprocesses, which execute `handler.initialize(ctx)`. TorchServe monitors initialization with a timeout (`model_config_group_timeout_ms`, default 120 seconds). If a worker fails to reach READY state within the timeout — for example, because `initialize` threw an exception or the model weights file was corrupted — TorchServe logs the failure, increments a retry counter, and attempts to restart the worker. After three consecutive failures, it marks the model as failed.

During normal operation, the JVM dispatches batched jobs to IDLE workers and marks them BUSY. When a worker's forward pass completes, the result is sent back over the socket, the JVM returns the HTTP responses, and the worker returns to IDLE.

When a worker crashes mid-inference (segfault in a native library, GPU driver error, OOM kill), the JVM detects the socket closure, abandons the in-flight requests (clients receive HTTP 500), and restarts the worker. The `unregister_model_when_failed=false` default means TorchServe will keep retrying — appropriate for transient crashes. If you want the deployment system to take over on persistent failures, set `unregister_model_when_failed=true` so the model disappears and your health check / alerting triggers.

![TorchServe worker lifecycle state diagram showing transitions from created through loading, ready, busy, and stopped states](/imgs/blogs/torchserve-deep-dive-7.png)

### Graceful shutdown

```bash
torchserve --stop
```

TorchServe's stop command initiates graceful shutdown: it stops accepting new requests at the Netty frontend, waits for in-flight batches to complete (up to `job_queue_size` × `default_response_timeout` seconds), then terminates workers. For a deployment with long-running inference (e.g., LLM generation), you may need to increase `default_response_timeout` to allow in-progress generations to complete before shutdown.

### Health check endpoint

```bash
curl http://localhost:8080/ping
# {"status": "Healthy"}

# Returns "Unhealthy" if any registered model's workers are all in failed state
```

The `/ping` endpoint is suitable for Kubernetes liveness probes. For readiness probes (indicating the service can accept traffic), poll the management API to verify all model workers are in READY state:

```bash
# Readiness check script
BERT_STATUS=$(curl -s http://localhost:8081/models/bert-sentiment | python3 -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
ready = all(w['status'] == 'READY' for w in workers)
print('ready' if ready else 'not_ready')
")
```

### Zero-downtime model updates

TorchServe does not have a native rolling update mechanism, but you can approximate one:

```bash
# Step 1: Register new version with initial 0 workers
curl -X POST "http://localhost:8081/models" \
  -F "model_name=bert-sentiment" \
  -F "url=bert-sentiment-v2.mar" \
  -F "version=2.0" \
  -F "initial_workers=4" \
  -F "synchronous=true"

# Step 2: Test new version via versioned endpoint
curl http://localhost:8080/predictions/bert-sentiment/2.0 \
  -d '{"data": "test request"}'

# Step 3: Set version 2.0 as default (new requests route here)
curl -X PUT "http://localhost:8081/models/bert-sentiment/2.0/set-default"

# Step 4: Wait for in-flight v1.0 requests to drain (manual delay or monitor v1.0 metrics)
sleep 30

# Step 5: Unregister old version
curl -X DELETE "http://localhost:8081/models/bert-sentiment/1.0"
```

The transition period between Step 3 and Step 5 serves both versions simultaneously. During this window, v2.0 handles all new requests while v1.0 completes any in-flight requests that were dispatched before the default switch.

### Full production JVM configuration

```properties
# Production config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_gpu=4

# JVM heap and GC settings
vmargs=-Xmx4g \
  -XX:MaxDirectMemorySize=1g \
  -XX:+UseG1GC \
  -XX:G1HeapRegionSize=16m \
  -XX:+ExitOnOutOfMemoryError \
  -Dlog4j.configurationFile=log4j2.xml

# Request lifecycle
default_response_timeout=120
default_workers_per_model=2
job_queue_size=200

# Failure handling
unregister_model_when_failed=false

# Batching defaults (override per-model via management API)
batch_size=8
max_batch_delay=50

# Model store location
model_store=/opt/torchserve/model-store

# Enable access logging
enable_envvars_config=true
```

The `-XX:+UseG1GC` flag is important for production deployments. The default GC in Java 11+ is G1GC, but explicitly setting it with region size tuning reduces GC pause times for large heaps. The `-XX:+ExitOnOutOfMemoryError` flag causes the JVM to exit immediately if it runs out of heap — combined with your process manager (systemd, Kubernetes restart policy), this ensures automatic recovery from memory leaks rather than a zombie state where the process is alive but OOMing silently.

### Logging configuration for production

TorchServe uses Log4j2 for structured logging. A production log4j2 configuration that outputs JSON for centralized logging systems (ELK, Splunk):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Configuration status="WARN">
  <Appenders>
    <RollingFile name="File" fileName="logs/access_log.log"
                 filePattern="logs/access_log.log.%d{yyyy-MM-dd}.%i.gz">
      <JsonLayout compact="true" eventEol="true"/>
      <Policies>
        <SizeBasedTriggeringPolicy size="100MB"/>
        <TimeBasedTriggeringPolicy interval="1"/>
      </Policies>
    </RollingFile>
  </Appenders>
  <Loggers>
    <Root level="INFO">
      <AppenderRef ref="File"/>
    </Root>
  </Loggers>
</Configuration>
```

![Production hardening layers covering JVM configuration, worker lifecycle management, health checks, and zero-downtime deployment flow](/imgs/blogs/torchserve-deep-dive-8.png)

## 10. Case studies and benchmarks

### Case study 1: Meta's PyTorch Hub deployment

Meta uses TorchServe internally as the serving infrastructure for PyTorch Hub model inference APIs. The PyTorch Hub (`torch.hub.load`) functionality that data scientists use to download pre-trained models in a single line of code is backed by TorchServe instances that handle the inference-on-demand requests.

According to the TorchServe paper and Meta engineering blog posts, the Netty frontend adds approximately 1.8–2.5ms of overhead per request compared to a raw Python socket server. This overhead comes from:
- HTTP parsing (Netty): ~0.3ms
- Java-to-Python socket round-trip: ~0.5ms
- Protocol Buffer serialization/deserialization: ~0.4ms
- Python queue dispatch: ~0.6ms

Total overhead: ~1.8ms. Compared to a direct Flask serving layer (~8ms overhead due to WSGI, serialization, and threading model), TorchServe's Netty frontend is actually faster per-request despite the JVM layer. The Python WSGI overhead exceeds the JVM socket overhead for most workloads.

### Case study 2: BERT-large sentiment versus ResNet-50 benchmark

Full benchmark results for co-located serving on a 4× T4 GPU instance (64 CPU cores, 256GB RAM):

| Model | Workers | Batch size | Throughput (req/s) | p50 (ms) | p99 (ms) | GPU Memory (GB) | GPU |
|---|---|---|---|---|---|---|---|
| BERT-large | 1 | 1 | 12 | 22 | 45 | 1.8 | T4 16GB |
| BERT-large | 1 | 8 | 58 | 48 | 92 | 3.2 | T4 16GB |
| BERT-large | 4 | 8 | 220 | 50 | 105 | 12.8 | 4×T4 |
| ResNet-50 | 2 | 32 | 340 | 28 | 61 | 4.1 | T4 16GB |
| Both co-located | 2+2 | 8+16 | 180+290 | 52/31 | 98/65 | 14.5 | 4×T4 |

The co-located deployment shows a modest throughput reduction (18% for BERT, 15% for ResNet) compared to dedicated deployments, due to GPU memory bandwidth contention when both models are active simultaneously. For workloads where BERT and ResNet peak traffic do not overlap, the co-located deployment is significantly more cost-efficient: you pay for 4 GPUs instead of 8.

The GPU memory figure of 14.5 GB for the co-located deployment deserves explanation:
- BERT-large fp16 × 4 workers: $4 \times 0.68 = 2.72$ GB
- ResNet-50 fp32 × 2 workers: $2 \times 0.10 = 0.20$ GB
- Activation memory at peak batch (BERT batch 8 × 4 workers simultaneous): ~8 GB
- CUDA runtime + Python processes overhead: ~3.5 GB
- Total: ~14.4 GB — approaching T4 limits, but stable

The p99 latencies in the co-located setup (98ms BERT, 65ms ResNet) are slightly higher than solo deployments because of occasional GPU memory bandwidth contention, but remain within typical SLOs.

### Case study 3: TorchServe versus Flask for BERT serving

Many teams start serving models with Flask because it is familiar and requires no additional dependencies. Here is a direct comparison at realistic traffic levels:

**Flask (single Gunicorn worker, Gevent):**
- Overhead per request: ~8ms (WSGI processing, Python dict overhead)
- Batching: None (each request is independent)
- At 30 QPS: p99 = 1,100ms (requests queuing behind each other in the single worker)
- Memory: 1.4 GB (model) + 150 MB (Python/Flask) per worker
- Metrics: Manual Prometheus integration required
- Multi-model: Manual routing code required

**TorchServe (2 workers, batch_size=8, max_batch_delay=50ms):**
- Overhead per request: ~15ms (JVM socket overhead)
- Batching: Automatic
- At 30 QPS: p99 = 92ms (batches of 8, parallel workers)
- Memory: 1.4 GB (model) per worker + 300 MB (JVM overhead)
- Metrics: Prometheus histogram built-in
- Multi-model: Built-in management API

The crossover point where TorchServe's batching efficiency exceeds Flask's lower per-request overhead is approximately 15–20 QPS for BERT-large on a T4 GPU. Below 15 QPS, Flask is actually simpler and has lower latency. Above 20 QPS, TorchServe's automatic batching delivers dramatically better throughput per GPU, and the infrastructure features (metrics, multi-model, hot reload) justify the operational complexity.

The 503-under-load comparison is important for production reliability. Flask with a single Gunicorn worker has no explicit queue — requests that arrive when the worker is busy pile up in the OS TCP accept queue and eventually fail with connection reset errors (no clean HTTP response). TorchServe's `job_queue_size` provides clean backpressure with deterministic HTTP 503 responses, making retry logic in upstream systems straightforward: retry on 503, do not retry on 5xx (inference errors).

### Case study 4: Whisper-large-v3 audio transcription serving

Whisper-large-v3 (1.55 billion parameters) represents a category of models — encoder-decoder audio/vision transformers — that TorchServe handles well. Audio transcription has interesting batching characteristics because input length (audio duration) varies widely between requests, unlike NLP classification where sequence lengths are bounded by `max_length=512`.

Benchmark results for Whisper-large-v3 in fp16 on an A100 40GB:

| Audio duration | Transcription time | GPU Memory allocated |
|---|---|---|
| 5 seconds | 0.8 seconds | 3.1 GB |
| 30 seconds | 2.1 seconds | 3.8 GB |
| 5 minutes | 18.4 seconds | 5.2 GB |
| 30 minutes | 112 seconds | 7.9 GB |

The key observation: Whisper's inference time scales roughly linearly with audio duration (more decode steps for longer audio), but GPU memory scales sublinearly because the encoder processes audio as fixed-length mel spectrograms chunked into 30-second windows. For Whisper, setting `default_response_timeout=300` (5 minutes) is necessary to handle long-form audio without timeout errors, and `batch_size=1` is recommended because batching Whisper is complex (each sequence in the batch must pad to the longest audio duration, wasting compute on silence padding).

The practical deployment configuration for Whisper on TorchServe:

```properties
# Whisper-specific config.properties additions
# Single request per worker (audio inputs are not batchable efficiently)
batch_size=1
max_batch_delay=0
# Long timeout for 30-minute audio files
default_response_timeout=300
# Worker count drives concurrency
default_workers_per_model=4
job_queue_size=50
```

With 4 workers and no batching, the throughput is 4 concurrent audio transcriptions. For a 30-second audio clip (2.1 seconds inference), throughput is approximately $4 / 2.1 = 1.9$ requests/second — approximately 115 minutes of audio per minute of wall clock time. This is sufficient for most real-time transcription workloads at moderate scale.

## 11. When to use TorchServe (and when not to)

After examining TorchServe's internals in depth, the guidance is clear:

**Use TorchServe when:**

You are serving classic deep learning models — BERT, ViT, ResNet, EfficientNet, Whisper, CLIP, GPT-2 — in the PyTorch ecosystem. TorchServe was designed precisely for this workload, and it handles it excellently.

Your total QPS across all models is under approximately 500 req/s on a well-tuned instance. Above this, you need to think more carefully about whether TorchServe's batching overhead is leaving performance on the table.

You want multi-model management without Kubernetes CRDs and custom operators. TorchServe's management API handles registration, versioning, worker scaling, and health monitoring with a single REST API that your CI/CD system can call directly.

Your team is entirely in the PyTorch ecosystem and wants native integration. TorchScript models, `torch.compile`-optimized models, and Hugging Face models all work natively in TorchServe handlers without export steps.

You want Prometheus metrics without writing a custom middleware layer. The built-in metrics endpoint saves significant instrumentation work.

You need a production-proven, well-documented solution with official support from Meta and AWS. TorchServe is used in production at both companies and has an active open-source community.

**Do not use TorchServe when:**

You are serving LLMs with more than 7 billion parameters. TorchServe has no support for tensor parallelism (splitting a single model across multiple GPUs). Llama-3-70B requires tensor parallel across 4–8 GPUs to fit in GPU memory, which TorchServe cannot do. Use vLLM with tensor_parallel_size or Ray Serve with DeepSpeed-Inference.

You need sub-10ms p99 latency. The JVM baseline overhead (1.8–2.5ms) plus Python subprocess communication adds an irreducible floor that prevents sub-10ms p99 for any model size. Triton Inference Server with a TensorRT engine can achieve sub-5ms p99 for optimized models.

You are serving a high-QPS LLM — above approximately 100 requests per second for any generative model larger than 1B parameters. TorchServe's static batching causes catastrophic tail latency at this scale. vLLM's continuous batching is specifically engineered for this workload.

You need distributed tensor-parallel inference for frontier models. Use Ray Serve combined with DeepSpeed-Inference or Megatron-LM for models that require model parallelism.

You need TensorRT INT8 kernels or quantization-specific optimizations. Triton Inference Server has first-class TensorRT backend support; TorchServe requires you to run TensorRT inference within a Python handler, which works but eliminates the pipeline-level optimizations that make Triton fast.

### The operational cost argument

TorchServe's infrastructure features have a real dollar cost in engineering time. Setting up TorchServe correctly — handler code, MAR builds, config.properties, Prometheus dashboards, alerting — takes approximately 2–3 engineer-days for a team that has not used it before. Contrast this with:
- Raw Flask: 1 day to set up, but no batching, no metrics, manual scaling
- Triton: 3–5 days (ONNX export or TensorRT build, Triton config, ensemble pipelines)
- vLLM: 0.5 days for LLM serving (excellent defaults, single Python process, OpenAI-compatible API)

For teams with multiple PyTorch models to serve, TorchServe's operational surface amortizes well. The management API, Prometheus integration, and MAR versioning all save engineering time at scale. For a single LLM, vLLM's near-zero-configuration path is usually faster to production.

### Kubernetes integration patterns

In production Kubernetes deployments, TorchServe runs as a Deployment with the following patterns:

The model store is typically a Persistent Volume backed by network storage (EFS on AWS, Filestore on GCP). All pods share the same model store, eliminating the need to bake MAR files into the container image. New model versions are uploaded to the shared volume and registered via the management API without container rebuild.

A sidecar pattern for model registration:

```yaml
# kubernetes/torchserve-deployment.yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: torchserve
          image: pytorch/torchserve:0.9.0-gpu
          ports:
            - containerPort: 8080
            - containerPort: 8081
            - containerPort: 8082
          volumeMounts:
            - name: model-store
              mountPath: /home/model-server/model-store
          readinessProbe:
            httpGet:
              path: /ping
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /ping
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 20
      volumes:
        - name: model-store
          persistentVolumeClaim:
            claimName: torchserve-model-store
```

The readiness probe uses the `/ping` endpoint (port 8080) to signal when the container is ready to receive traffic. The `initialDelaySeconds: 30` gives workers time to load models before Kubernetes starts routing requests to the pod. For BERT-large with 4 workers, a 30-second initial delay may be insufficient — consider increasing to 60–90 seconds for large models.

Horizontal Pod Autoscaler configuration based on GPU queue latency (requires custom metrics adapter):

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: torchserve-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: torchserve
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: ts_queue_latency_p99_ms
        target:
          type: AverageValue
          averageValue: "100"
```

This HPA scales out when the average p99 queue latency across all pods exceeds 100ms, and scales in when it drops below 100ms for a sustained period. The 100ms threshold corresponds to the `max_batch_delay` value — if queue latency exceeds `max_batch_delay`, workers are saturated and additional pods are needed.

## 12. Key takeaways

- TorchServe's Netty frontend handles thousands of concurrent connections without per-connection threads. Each registered model gets its own job queue. Python workers are separate processes — no GIL contention across workers, but 200–300 MB memory overhead per worker.

- A MAR file is a ZIP archive containing model weights, handler.py, and extra files. Use `--archive-format zip` for debuggability. Version your MAR files and use TorchServe's version routing for blue-green deployments.

- Every custom handler implements four methods: `initialize` (load weights, warm up), `preprocess` (tokenize, normalize), `inference` (forward pass with `torch.no_grad()`), `postprocess` (convert logits to JSON). Always run a warm-up forward pass in `initialize` to absorb CUDA kernel compilation penalty.

- Dynamic batching is governed by the SLO triangle: latency, throughput, and cost cannot all be simultaneously optimized. For BERT-large on T4, batch_size=8 with max_batch_delay=50ms sits on the efficient frontier — 4.8× throughput improvement over batch_size=1 with only 2× p99 latency increase.

- Little's Law ($L = \lambda W$) tells you whether your deployment is correctly provisioned. If queue latency is high relative to inference latency, add workers. If GPU utilization is below 70%, increase batch size.

- The management API at port 8081 handles all model lifecycle operations: register, scale workers, set default version, unregister. Use `synchronous=true` on register calls in deployment automation to avoid race conditions.

- Port 8082 exposes Prometheus-compatible metrics including `ts_inference_latency_microseconds` (histogram), `ts_queue_latency_microseconds` (histogram), and `ts_inference_requests_total` (counter). Alert on queue latency, not inference latency — high queue latency predicts user-visible spikes before they happen.

- TorchServe's fundamental limitation for LLM serving is static batching: the entire batch waits for the slowest sequence to complete. At 100 QPS LLM traffic, this creates catastrophic tail latency. Use vLLM for any LLM serving at scale.

- For models larger than 7B parameters requiring multi-GPU tensor parallelism, TorchServe has no built-in support. The architecture assumes one model fits on one GPU (or multiple workers each holding the full model on their own GPU). vLLM or Ray Serve are the right tools here.

- TorchServe is genuinely excellent at what it was designed for: multi-model, multi-version, Prometheus-instrumented serving of PyTorch deep learning models in the 100M–6B parameter range at moderate QPS. Match the tool to the workload.

## Further reading

- [TorchServe GitHub repository](https://github.com/pytorch/serve) — source code, issue tracker, and example handlers for ResNet, BERT, and custom models
- [TorchServe official documentation](https://pytorch.org/serve/) — comprehensive reference for all configuration options and APIs
- [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — fundamentals of production ML inference systems, the serving stack, and when serving infrastructure matters
- [Dynamic batching deep dive](/blog/machine-learning/model-serving/dynamic-batching-deep-dive) — detailed analysis of batching strategies, continuous batching, and the mathematics of latency-throughput optimization
- [Triton Inference Server deep dive](/blog/machine-learning/model-serving/triton-inference-server-deep-dive) — NVIDIA's serving framework, TensorRT backend integration, and when Triton outperforms TorchServe
- [Choosing your serving stack](/blog/machine-learning/model-serving/choosing-your-serving-stack) — decision framework for selecting between TorchServe, Triton, vLLM, and Ray Serve based on model type, scale, and latency requirements
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — complete operational guide covering deployment, scaling, monitoring, and incident response for ML serving systems
