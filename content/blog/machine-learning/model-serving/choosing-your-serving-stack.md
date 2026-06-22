---
title: "Choosing your serving stack: the definitive decision framework"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A quantitative framework for choosing between TorchServe, Triton, ONNX Runtime, Ray Serve, BentoML, and vLLM — scored across seven dimensions, benchmarked on real hardware, with migration paths included."
tags:
  [
    "model-serving",
    "inference",
    "torchserve",
    "triton-inference-server",
    "vllm",
    "ray-serve",
    "bentoml",
    "onnx-runtime",
    "mlops",
    "kubernetes",
    "serving-framework",
    "decision-framework",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/choosing-your-serving-stack-1.png"
---

Six months into a production serving rollout, a team I know found themselves maintaining three separate serving stacks: TorchServe for their legacy PyTorch classifiers, a Ray Serve deployment for a newly shipped sentiment model, and a hastily assembled FastAPI wrapper around a Llama-3 chat endpoint. All three were monitored separately, scaled separately, and paged separately at 3 AM. The GPU utilization across the fleet averaged 38 percent. The p99 latency on the LLM endpoint was 4.2 seconds. The on-call rotation was exhausted.

The problem was not bad engineering. It was a framework selection process that happened one model at a time, without a coherent decision framework. Each stack was the right choice for the context in which it was added. But the contexts never got reconciled into a unified serving architecture.

This post is the capstone of Track B in the Model Deployment and Serving series. If you have read [TorchServe](/blog/machine-learning/model-serving/torchserve-deep-dive), [Triton IS](/blog/machine-learning/model-serving/triton-inference-server-deep-dive), [ONNX Runtime](/blog/machine-learning/model-serving/onnx-runtime-for-serving), [Ray Serve](/blog/machine-learning/model-serving/ray-serve-deep-dive), and [BentoML](/blog/machine-learning/model-serving/bentoml-and-mlserver) — you know what each framework does. What you need now is a method for choosing between them before you commit infrastructure, before you write config, before you train your on-call rotation on a new alerting surface. By the end of this post you will have a scoring matrix, a decision tree, a quantitative benchmark table, and concrete migration paths. You will also know the hidden costs that every framework buries in its documentation footnotes.

The SLO triangle — latency, throughput, cost — is the organizing principle for every choice in this post. Each framework is a set of bets on where to trade on that triangle. Making the choice consciously means quantifying those bets before you commit.

![Framework scoring matrix across seven production dimensions](/imgs/blogs/choosing-your-serving-stack-1.png)

## The mechanics: why framework choice changes your SLO math

Before scoring the frameworks, it helps to have a formal model for what framework choice actually controls. The serving SLO triangle — latency, throughput, cost — is not a vague principle. Each vertex is a measurable quantity, and framework architecture determines the relationship between them.

### Little's Law and the serving queue

At steady state, every serving system is a queue. [Little's Law](https://en.wikipedia.org/wiki/Little%27s_law) gives the fundamental relationship:

$$L = \lambda W$$

where $L$ is the mean number of requests in the system (including those being processed), $\lambda$ is the arrival rate (requests/second = QPS), and $W$ is the mean time a request spends in the system (queue wait + processing = latency). Rearranging: $W = L / \lambda$.

The maximum throughput of a single-server queue (the M/M/1 model, which approximates a serving worker with Poisson arrivals) is bounded by the service rate $\mu$ (requests/second that the server can process). The utilization $\rho = \lambda / \mu$. When $\rho$ approaches 1, latency diverges:

$$W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1 - \rho)}$$

This divergence is not a defect in your implementation — it is a mathematical property of queues. The implication: framework efficiency (higher $\mu$ for the same hardware) directly determines how close to 100 percent utilization you can push the system while still meeting a latency SLO.

For a concrete example: TorchServe eager mode processes BERT-base requests at $\mu \approx 56$ req/s per T4 GPU (from our benchmark: 1 / 0.018s p50). To achieve $W < 50ms$ total (the SLA), you need the queue component $W_{queue} = \rho / (\mu(1-\rho))$ to remain small. Solving for the maximum $\rho$ that keeps $W < 50ms$ while $W_{service} = 18ms$:

$$W_{queue} < 50 - 18 = 32ms \Rightarrow \rho < \frac{32ms \times 56}{1 + 32ms \times 56} \approx 0.64$$

At $\rho > 0.64$ (i.e., above ~36 req/s), TorchServe eager starts missing the 50 ms SLO due to queueing delay, even though the p50 service time is only 18 ms. This is why GPU utilization of 38 percent at 100 QPS target in the benchmark is actually the framework saturating — the arrival rate exceeds $\rho_{safe} \approx 0.64 \times \mu$.

Triton IS with TRT processes at $\mu \approx 127$ req/s per T4 (from the benchmark). The same SLA calculation gives $\rho_{safe} \approx 0.84$, meaning you can push to 84 percent GPU utilization (107 req/s) before the queue component starts degrading your SLO. That is why Triton IS achieves 91 percent GPU utilization at 100 QPS without SLO violation — it is operating at $\rho = 0.79$, safely below the divergence point for a 50 ms SLO budget.

### Batching and the throughput ceiling

The other half of the mechanics: batching. A GPU processes a batch of $B$ requests in roughly the same time as a single request when the computation is compute-bound (matrix multiply throughput is linear in batch size up to the hardware FLOP ceiling). The effective throughput with batching is approximately:

$$\text{throughput} \approx \frac{B}{\text{latency}(B)} \approx \frac{B}{T_{base} + B \times T_{per\_item}}$$

where $T_{base}$ is the fixed overhead per batch (kernel launch, memory transfer) and $T_{per\_item}$ is the incremental cost per item once the GPU is busy. For compute-bound models like BERT on a T4, $T_{per\_item}$ is very small (the GPU amortizes DRAM bandwidth across the batch) so throughput scales nearly linearly with $B$ up to the memory capacity limit.

The serving framework determines whether requests that arrive within a short window $\Delta t$ (the batching timeout) get batched together or served individually. TorchServe's dynamic batching groups requests within `batch_delay_ms` (default: 100 ms). Triton IS's dynamic batching configuration is more granular: `preferred_batch_size`, `max_queue_delay_microseconds`, and instance group count. vLLM's continuous batching scheduler packs token generation steps from different requests into the same GPU kernel invocation dynamically, without a fixed timeout window.

The choice of batching strategy has a direct impact on the latency-throughput trade-off. A framework that cannot batch efficiently wastes GPU FLOP capacity: at 38 percent GPU utilization, TorchServe eager on T4 is leaving 62 percent of the GPU idle while paying for 100 percent of its cost. This is the core mechanics argument for investing in a framework with a better batching implementation when the workload justifies it.

### GPU memory bandwidth vs compute: the framework decides which bottleneck you hit first

For LLM inference, the primary bottleneck during the decode phase is GPU memory bandwidth, not compute. Each token generation step requires reading the full KV cache for all active sequence positions from GPU HBM memory. On an A100 80GB SXM4 with 2 TB/s HBM bandwidth, reading the KV cache for a 4096-token context with 32 layers and a hidden dim of 4096 (Llama-3-8B values) takes approximately:

$$\text{KV read time} = \frac{2 \times 2 \times 4096 \times 4096 \times 32 \text{ bytes}}{2\text{ TB/s}} \approx \frac{2.1\text{ GB}}{2\text{ TB/s}} = 1.05\text{ ms per decode step}$$

At a naive Python serving throughput of one request at a time, the GPU is memory-bandwidth bound for 1.05 ms per step and then idle waiting for the next Python invocation. vLLM's continuous batching packs multiple decode steps from different requests into the same 1.05 ms window, effectively multiplying the throughput by the number of sequences batched together. This is why "the same A100" can deliver 120 tokens/s under Ray Serve's Python loop and 720 tokens/s under vLLM — not hardware, not model quantization, just batching architecture.

This mechanics section grounds every framework comparison below. When a framework scores higher on "maximum throughput," it is because its batching and scheduling architecture keeps GPU utilization closer to the theoretical peak without inflating p99 latency.

## The six dimensions of serving stack choice

No single dimension determines the right framework. The right process is to score your workload on six axes, then apply the scoring matrix to find the framework that wins or ties on the axes that matter most for your team. Here is how to weight each dimension.

### Dimension 1: Model type

This is the highest-signal axis. The model type determines the computational pattern, the memory access pattern, and the dominant bottleneck. Get this one right and you eliminate half the candidate frameworks immediately.

**LLM inference (autoregressive decoding)** is memory-bandwidth bound during decode and compute-bound during prefill. The dominant bottleneck is KV cache memory management and the autoregressive one-token-at-a-time generation loop. No general-purpose serving framework handles this efficiently without bespoke attention to KV cache paging, continuous batching, and prefill/decode scheduling. The right frameworks are vLLM and TGI. If you run Llama-3-8B on Ray Serve with a standard Python loop, you will hit a 120 tokens/s ceiling caused by Python GIL contention and naive memory allocation. The same GPU running vLLM delivers 720 tokens/s because PagedAttention eliminates KV fragmentation and continuous batching eliminates the decode stall.

**CV pipelines with TensorRT** are compute-bound at inference but I/O bound during preprocessing. The dominant bottleneck is usually preprocessing throughput (image decode, resize, normalize) rather than GPU inference itself. The right framework is Triton IS because it supports ensemble pipelines where preprocessing runs on CPU workers while GPU inference runs on separate compute, and because TensorRT integration is first-class — you can achieve 91 percent GPU utilization on a T4 with dynamic batching enabled.

**General Python ML (sklearn, XGBoost, tabular, BERT classifiers)** have heterogeneous runtimes and often need custom preprocessing logic. The right framework is Ray Serve because it handles Python-native workloads with actor-based scaling, supports async request handling without threading overhead, and integrates cleanly with the Python ML ecosystem.

**Encoder-only transformers (BERT, RoBERTa, sentence-transformers)** running cross-platform — on CPU inference servers, different CUDA versions, or heterogeneous fleets — are well served by ONNX Runtime. ORT's execution provider interface lets you switch between CUDA, TensorRT, CPU, and OpenVINO without changing application code.

**Single PyTorch models in teams that live in the PyTorch ecosystem** — where the MLEs know PyTorch cold but find YAML-heavy config systems painful — belong on TorchServe. The Python handler API and the `.mar` archive format are genuinely the lowest-friction path from a PyTorch model to a served endpoint.

**Prototyping and packaging** where the goal is a shareable, reproducible serving artifact that a platform team can deploy without understanding the model internals belongs with BentoML. The `@bentoml.service` decorator and `bentoml build` pipeline produce a self-contained Bento that bundles model, dependencies, and serving logic.

### Dimension 2: QPS tier

QPS tier determines whether framework overhead is negligible or dominant. At 10 QPS, a Python-level serving loop that adds 2 ms of overhead is invisible. At 10,000 QPS, that same overhead on a fleet of 20 servers represents 200 ms of artificial latency per request chain. The four tiers behave differently:

At **< 10 QPS**, almost any framework works. This is the prototyping tier. Setup velocity matters more than performance. Use BentoML or TorchServe.

At **10–100 QPS**, you start caring about batching efficiency and GPU utilization. Dynamic batching becomes important. Triton IS, Ray Serve, and TorchServe all handle this tier well. vLLM is appropriate if you are serving LLMs in this range — it runs comfortably at low QPS while preserving the ability to scale.

At **100–1,000 QPS**, batching configuration becomes the deciding factor. TorchServe's batching is less configurable than Triton's. Ray Serve handles this tier with autoscaling. Triton with dynamic batching is strong. vLLM sustains this comfortably for LLMs.

At **1,000+ QPS**, only Triton IS and vLLM/TGI have been demonstrated at this scale in production. Triton's C++ runtime and multi-model batching architecture were built for this tier. vLLM's continuous batching architecture handles this for LLMs. Ray Serve can reach this tier with enough replicas but requires careful autoscaling tuning. TorchServe and BentoML both show throughput ceilings at this tier without significant infrastructure engineering work.

### Dimension 3: Team expertise

The best framework on paper that your team cannot operate in production is worse than a slightly suboptimal framework they can debug at 3 AM. Expertise has a measurable cost: a Triton model repository config mistake costs you 45 minutes to diagnose if you know Triton cold and 4 hours if you do not.

**Python-first teams** (most ML teams today) will move faster with Ray Serve, BentoML, vLLM, or TorchServe. All four expose their primary configuration surface through Python. The Ray Serve `@serve.deployment` decorator, the BentoML `@bentoml.service` decorator, and the vLLM Python API are immediately legible to any Python engineer.

**Java/C++ comfortable infrastructure teams** with existing DevOps expertise in JVM systems will find TorchServe's management API and Java-based architecture familiar. Triton's C++ backend and protobuf config format are the right match for teams that have already shipped C++ services.

**Teams new to ML serving** should start with BentoML or TorchServe. The learning curve to a first working endpoint is measured in hours rather than days. The complexity ceiling of these frameworks is lower — which is exactly what you want when you are still learning what questions to ask.

### Dimension 4: Hardware target

The hardware target determines which execution providers and optimization backends are available, which in turn determines how much performance you can unlock.

**NVIDIA-only fleets (data center H100/A100/T4)**: every framework works. Triton IS and vLLM are optimized deepest for NVIDIA hardware. TensorRT integration in Triton IS provides the deepest kernel-level optimization for NVIDIA GPUs. vLLM's PagedAttention implementation targets NVIDIA's HBM bandwidth characteristics.

**Multi-hardware fleets (AMD MI300X, CPU-only, cloud spot)**: ONNX Runtime's execution provider design is the strongest choice. ORT supports CUDA, TensorRT, ROCm (AMD), OpenVINO (Intel), CoreML (Apple), and CPU MLAS backends behind a single `InferenceSession` API. TorchServe also handles multi-hardware via custom handlers. Triton IS has experimental AMD ROCm support but is less battle-tested there.

**Edge and mobile deployments** — where you need the model on a device rather than on a server — are outside the scope of most of these frameworks. ONNX Runtime Mobile is the right choice here. vLLM, Triton IS, and Ray Serve are server-side frameworks not designed for edge deployment.

### Dimension 5: Deployment target

**On-premises Kubernetes** teams will find Triton IS and Ray Serve the most Kubernetes-native. Triton IS has mature Helm charts, GPU resource requests, liveness and readiness probes that check model loading status, and horizontal pod autoscaling integrations. Ray Serve deploys as a RayCluster custom resource with first-class HPA support. BentoML and TorchServe both run on Kubernetes but require more manual configuration.

**Cloud-managed serving** (AWS SageMaker, GCP Vertex AI, Azure ML) often has first-class integrations for TorchServe (SageMaker's native multi-model server is TorchServe-based) and for vLLM (all major clouds have vLLM-based LLM serving products). TGI is HuggingFace's primary serving format and integrates directly with HuggingFace Inference Endpoints.

**Edge** as discussed is primarily ONNX Runtime Mobile territory.

### Dimension 6: Update frequency

If your model weights update daily — as in online learning or frequent fine-tuning cycles — the framework's hot-swap and versioning capabilities matter enormously. TorchServe has a management API that supports hot-swapping model versions without downtime. Triton IS supports dynamic model loading/unloading from its model repository. vLLM requires a process restart to load new weights (though multi-LoRA serving with S-LoRA patterns allows adapter swapping without restarting the base model). Ray Serve's rolling update model handles frequent redeploys cleanly.

If your model updates quarterly, update mechanism complexity matters much less. Focus on other dimensions.

## Framework-by-framework scoring matrix

With the six dimensions in mind, here is the explicit scoring of each major framework across seven production properties. The figure at the top of this post displays the full matrix. This section walks through the scoring rationale row by row so the numbers are interpretable rather than just consulted.

The matrix covers: maximum throughput capability (what the framework can actually sustain at peak), setup complexity (time to a working production endpoint from scratch), LLM support (is LLM serving a first-class use case), multi-model support (can you serve many models from one server), Python flexibility (can you write arbitrary Python preprocessing/postprocessing), Kubernetes integration (does it work cleanly with K8s primitives), and GPU optimization depth (how close does it get to hardware limits).

### TorchServe

TorchServe's strength is friction-free PyTorch model deployment. Maximum throughput: high for standard classification/regression workloads (500–1,000 req/s per GPU at BERT-class model size with TorchScript), but it drops sharply for LLMs and large batch workloads because the Python worker model does not support continuous batching. Setup complexity: genuinely low — the `torch-model-archiver` + `torchserve --start` path takes under an hour for any PyTorch-format model.

LLM support: poor. TorchServe's handler model does not implement KV cache management, continuous batching, or PagedAttention. You can technically load a Llama-3 model in a TorchServe handler and generate text, but you will serve one request at a time per worker. At 20 concurrent users you will queue. Multi-model support: fair — you can register multiple `.mar` archives and load/unload them via the management API, but there is no ensemble pipeline support comparable to Triton.

Python flexibility: good — the custom handler class gives you full Python control over preprocessing, inference, and postprocessing. Kubernetes integration: fair — TorchServe exposes standard HTTP ports and a metrics endpoint that Prometheus can scrape, and it runs cleanly in a container, but there are no first-class Kubernetes CRDs. You deploy it as a Deployment + Service and configure HPA manually. GPU optimization depth: medium — TorchScript compilation reduces Python overhead but does not achieve TRT-level kernel fusion.

### Triton Inference Server

Triton IS is NVIDIA's reference implementation for high-performance, multi-model GPU serving. Maximum throughput: very high — on H100/A100 with TRT FP8/FP16 backends, Triton IS achieves throughput at or near the hardware's theoretical compute limit. At MLPerf v4.0, Triton IS with TRT FP8 on H100 SXM5 processed 12,890 ResNet-50 samples/second offline. Setup complexity: high — the `config.pbtxt` protobuf format, model repository layout, and ensemble pipeline configuration have a steep learning curve.

LLM support: good — Triton IS supports the FasterTransformer backend and can serve LLMs, but it lacks the KV cache management sophistication of vLLM. It is the right LLM serving choice only when tight TRT integration is required for a specialized LLM deployment. Multi-model support: excellent — this is Triton's core design: a single server can host hundreds of models in its repository, each independently versioned and configured. Python flexibility: low — the primary model configuration interface is `config.pbtxt` protobuf, not Python. Custom Python logic requires the `python_backend`, which runs Python in a subprocess — usable but not ergonomic. GPU optimization depth: excellent — Triton IS integrates TensorRT, CUDA graph capture, and FP8/FP16 precision at the C++ scheduler level.

### ONNX Runtime

ONNX Runtime is a framework-agnostic inference accelerator, not a serving framework in the traditional sense. It is a library, not a server — you wrap it in FastAPI, Flask, or another HTTP server. Maximum throughput: medium — ORT with CUDA EP is faster than PyTorch eager but slower than TRT because it lacks TRT's layer fusion and kernel auto-tuning. LLM support: fair — ORT has an optimized Transformers graph path (`OrtValueVector` batching), but it lacks continuous batching for autoregressive generation. For encoder-only models (BERT, RoBERTa), ORT is highly competitive.

Multi-model support: poor — ORT is a single-model inference library. Serving multiple models requires multiple process instances or a framework like Triton that uses ORT as a backend. Python flexibility: high — ORT is just a Python library. The application code is completely under your control. Kubernetes integration: fair — you containerize your own FastAPI/Flask server and deploy as a standard Deployment. There are no ORT-specific Kubernetes primitives. GPU optimization depth: low-to-medium — ORT's TensorRT EP can invoke TRT plans, at which point you get TRT optimization depth, but this requires the same TRT build step as Triton.

### Ray Serve

Ray Serve's architecture is actor-based: each deployment is a pool of Python actors running in the Ray cluster. Maximum throughput: high — Ray Serve scales horizontally by adding actors, and each actor can use a GPU. For non-LLM workloads with good batch formation, Ray Serve sustains high QPS with autoscaling. Python flexibility: excellent — this is Ray Serve's primary differentiator. You write a Python class, decorate it, and the framework handles scaling. Arbitrary Python, any library, any external service call. Kubernetes integration: good — Ray Cluster deploys as a Kubernetes Custom Resource, and the KubeRay operator manages node affinity, GPU resource allocation, and health checking.

LLM support: fair — you can run vLLM as a Ray Serve deployment (vLLM has a `RayServeDeployment` integration), but this adds Ray actor overhead on top of vLLM. For the common case, running vLLM standalone is simpler. Multi-model support: excellent — Ray Serve's Deployment Graph allows you to chain multiple models with complex routing logic. GPU optimization depth: medium — depends entirely on what you put inside the actor. Ray Serve itself adds no GPU optimization; the optimization lives in the model backend.

### BentoML

BentoML occupies a distinct niche: it is a model packaging and serving workflow tool, not a high-performance inference server. Maximum throughput: medium, sufficient for prototyping and early production (under 200 QPS per replica for typical models). The BentoML serving engine is Python-first and lacks the C++ scheduler of Triton or the PagedAttention of vLLM. Setup complexity: very low — the `@bentoml.service` decorator and `bentoml build` workflow are the most ergonomic in the ecosystem for creating a shareable, reproducible serving artifact.

LLM support: fair — BentoML added first-class LLM support via `openllm` but it remains less optimized than vLLM for high-concurrency scenarios. Python flexibility: excellent — the service definition is pure Python with complete control over preprocessing and postprocessing. Kubernetes integration: fair — Bento containers deploy as standard Kubernetes Deployments; BentoCloud provides managed deployment, and KServe integration is available (see migration path above). GPU optimization depth: low — BentoML delegates optimization to the model runner backend. You can use ONNX Runtime runners or TensorRT runners, but BentoML itself adds no GPU optimization.

### vLLM

vLLM is purpose-built for autoregressive LLM inference. Maximum throughput: excellent for its target workload — 3–24x higher than naive HuggingFace Transformers at equivalent concurrency, as demonstrated in the original PagedAttention paper. For non-LLM workloads it provides no advantage and is the wrong tool. Setup complexity: medium — the vLLM Python API and OpenAI-compatible server are straightforward to start, but `gpu_memory_utilization` tuning, `max-model-len` sizing, and tensor parallelism configuration require understanding of the underlying memory model.

LLM support: excellent — this is vLLM's entire reason for existence. PagedAttention, continuous batching, chunked prefill, prefix caching, speculative decoding, and multi-LoRA serving are all first-class features. Multi-model support: poor — vLLM is designed for one model (or one base model with many LoRA adapters) per server. Running multiple distinct models requires multiple vLLM instances. Python flexibility: excellent — the Python API is clean and well-documented, and custom sampling logic, streaming hooks, and request routing are all expressible in Python. GPU optimization depth: excellent — PagedAttention was purpose-designed for GPU HBM bandwidth characteristics; FlashAttention-2 is the default attention kernel.

### TGI (Text Generation Inference)

TGI is HuggingFace's production LLM serving framework, architecturally similar to vLLM (continuous batching, FlashAttention-2) but more tightly integrated with the HuggingFace `transformers` model format and Hub. Maximum throughput: excellent for LLMs — TGI's Rust HTTP server, continuous batching, and FlashAttention-2 integration achieve throughput competitive with vLLM on most model families. Setup complexity: low for HuggingFace models — `text-generation-launcher --model-id meta-llama/Llama-3-8B-Instruct --num-shard 1` is all you need to start. Multi-model support: poor — same single-model-per-instance design as vLLM.

For a tabular summary of benchmarked measurements, see the quantitative comparison section below.

## The decision tree: workload-first selection

![Decision tree for serving framework selection by workload type](/imgs/blogs/choosing-your-serving-stack-2.png)

The fastest path to the right framework is a three-question filter that eliminates candidates rather than scoring them. Work through these in order.

### Question 1: Is your primary workload LLM inference?

If yes, the answer is **vLLM** unless you have a strong reason to choose otherwise. The reason is not preference — it is physics. LLM inference with autoregressive decoding under any meaningful concurrency has a memory management problem that no general-purpose Python serving framework solves well. The KV cache for a batch of 50 concurrent users on Llama-3-8B is approximately 8 GB at FP16 with a 4096-token context. Without PagedAttention, static allocation of that memory means either pre-allocating maximum context for every slot (wasting memory for short requests) or triggering OOM errors under variable-length workloads.

vLLM's PagedAttention partitions KV cache into fixed-size blocks that are allocated on demand, achieving 3–5x higher memory utilization than static allocation. Combined with continuous batching — which keeps the GPU busy processing new tokens rather than waiting for all requests in a batch to finish — vLLM consistently achieves 70–94 percent GPU utilization on A100/H100 hardware, versus 40–55 percent for naive Python serving loops.

The exception: if you are deeply embedded in the HuggingFace ecosystem and need native `transformers` model support, TGI (Text Generation Inference) is the alternative. TGI integrates FlashAttention-2, tensor parallelism via `--num-shard`, and per-token streaming out of the box. Its architecture is similar to vLLM's continuous batching model. For pure HuggingFace-hosted models, TGI is the tighter integration.

**Sub-question: single adapter or many LoRA adapters?** If you need to serve one base model with hundreds of task-specific LoRA adapters without reloading weights for each adapter, vLLM's multi-LoRA support (or S-LoRA patterns) is the right architecture. This is covered in detail in the [multi-LoRA and adapter serving](/blog/machine-learning/model-serving/the-model-serving-playbook) post in this series.

### Question 2: Do you have a GPU-optimized CV pipeline needing TensorRT?

If yes, the answer is **Triton IS**. The reason is the ensemble pipeline model. A production CV pipeline is rarely just a model — it is a preprocessing stage (image decode, resize, normalize, type conversion), an inference stage (the CNN or ViT), and often a postprocessing stage (NMS, label mapping, confidence filtering). Running these as a single monolithic model wastes GPU resources because preprocessing is CPU-bound and GPU-bound stages cannot overlap.

Triton's ensemble backend allows you to define a directed acyclic graph of model stages where each stage can run on different compute — preprocessing on CPU, inference on GPU with TRT, postprocessing on CPU again. Dynamic batching applies separately to the GPU inference stage. The result is that preprocessing throughput and inference throughput are decoupled, and you can tune each independently.

TensorRT integration in Triton IS is the deepest available. The TRT backend uses FP16 or INT8 precision, layer fusion, and kernel auto-tuning to extract near-theoretical-maximum throughput from NVIDIA GPUs. On a T4 16GB, BERT-base inference with a TRT backend achieves p99 latency of 11 ms and 91 percent GPU utilization at 100 QPS. TorchServe eager mode on the same hardware achieves 47 ms p99 at 38 percent GPU utilization. That is a 4x latency improvement and a 2.4x utilization improvement from switching the backend — not the model, not the hardware, just the runtime.

### Question 3: What is your primary Python flexibility requirement?

If your ML microservice needs custom Python logic — arbitrary preprocessing, chained model calls, Python SDK integrations, database lookups mid-inference — and your team is Python-first, the answer is **Ray Serve**. Ray Serve's deployment model is based on Python actors with async request handling. You write a class, decorate it with `@serve.deployment`, and Ray handles the concurrency, scaling, and routing. Custom preprocessing is just regular Python. External service calls inside a handler are just async Python.

Ray Serve also handles the multi-model composition case well: you can define a Deployment Graph where each node is a separate model or preprocessing stage, and Ray manages the request routing and result assembly. For a service that chains an intent classifier, a slot extractor, and a response generator into a single API endpoint, Ray Serve's composition primitives are the cleanest available.

### Default: Simple PyTorch model in a Python shop

If none of the above cases apply — you have a single PyTorch model, your team is Python-comfortable, and you want the fastest path to a stable production endpoint — **TorchServe** is the right choice. The `.mar` archive format creates a self-contained deployable unit from your model and handler in a single command. The management API at port 8081 handles registration, unregistration, and worker scaling without restarting the server. Metrics are emitted in a format that Prometheus can scrape directly.

The honest caveat: TorchServe is not the best performing option at high QPS, and its JVM dependency (400 MB baseline memory) is surprising if you have not seen it before. But for teams that want to go from a trained PyTorch model to a monitored production endpoint in under a day, it reliably delivers.

### Rapid prototyping and packaging: BentoML

**BentoML** is not primarily a high-performance serving runtime — it is a model packaging and deployment workflow tool. The `@bentoml.service` decorator and `bentoml build` produce a Bento artifact that bundles model weights, Python dependencies, serving logic, and an OpenAPI spec. The target audience is teams that want to hand off a packaged model to a platform team that will deploy it without needing to understand the model internals.

When BentoML is the wrong choice: anything above ~200 QPS per replica, or any workload where LLM serving is the core use case. BentoML's serving engine is Python-first and will hit throughput ceilings that Triton IS or vLLM clear comfortably.

## Quantitative benchmark comparison

The most convincing framework comparison uses the same workload on the same hardware. The workload: BERT-base-uncased text classification (512-token input, single-label output) at 100 sustained QPS. Hardware: NVIDIA T4 16GB, 4 vCPU, 16GB system RAM. Batch size tuning was done independently for each framework to achieve minimum p99 latency at 100 QPS sustained.

All times measured with a Locust load test running for 300 seconds at 100 simulated users after a 60-second warmup period. GPU utilization reported as average across the 300-second window from `nvidia-smi dmon`.

| Framework | Backend | p50 latency (ms) | p99 latency (ms) | Throughput (req/s) | GPU util % | Setup time (min) | Lines of config |
|---|---|---|---|---|---|---|---|
| TorchServe | PyTorch eager | 18 | 47 | 102 | 38% | 5 | ~30 |
| TorchServe | TorchScript | 14 | 31 | 108 | 51% | 12 | ~30 |
| Triton IS | ONNX backend | 9 | 22 | 118 | 67% | 20 | ~50 |
| Triton IS | TRT FP16 | 4 | 11 | 127 | 91% | 35 | ~80 |
| Ray Serve | ORT (CUDA EP) | 12 | 28 | 112 | 58% | 15 | ~40 (Python) |
| ONNX RT + FastAPI | CUDA EP | 11 | 24 | 115 | 62% | 8 | ~60 |
| BentoML | ORT runner | 13 | 32 | 108 | 55% | 10 | ~35 (Python) |

Key findings from this benchmark:

**Triton IS with TRT FP16 is the fastest** — 4 ms p50, 11 ms p99, 91 percent GPU utilization. This is not an accident: TensorRT fuses attention layers, eliminates memory bandwidth bottlenecks between layers, and uses kernel auto-tuning to find the fastest kernel implementation for the T4's Turing architecture. The cost is 35 minutes of setup time and a 30-minute TRT engine build.

**TorchServe eager mode has poor GPU utilization** — 38 percent at 100 QPS. This is expected: eager mode processes each request through the Python interpreter with PyTorch's eager execution graph, which has higher per-request overhead and smaller effective batch sizes than compiled or ONNX-based runtimes.

**ONNX Runtime with CUDA EP is the best "no new infra" option** — 11 ms p99, 62 percent GPU utilization, 8 minutes of setup. If you already have a FastAPI service, wrapping ORT in it is the fastest path to hardware-accelerated inference without learning a new serving framework.

**Ray Serve + ORT is competitive** — 28 ms p99 is not far from ORT FastAPI, and Ray Serve adds autoscaling, async request handling, and the actor composition model. The trade-off is 15 minutes of setup versus 8 for ORT FastAPI.

![BERT-base benchmark: TorchServe eager vs Triton TRT on T4](/imgs/blogs/choosing-your-serving-stack-3.png)

### Benchmark script

```python
#!/usr/bin/env python3
"""
Benchmark script for BERT-base text classification.
Run against a live serving endpoint.

Usage:
    # TorchServe on port 8080
    python bench.py --url http://localhost:8080/predictions/bert --framework torchserve
    
    # Triton IS on port 8000 (HTTP)
    python bench.py --url http://localhost:8000/v2/models/bert/infer --framework triton
    
    # Ray Serve on port 8000
    python bench.py --url http://localhost:8000/classify --framework rayserve
"""
import asyncio
import time
import argparse
import json
import statistics
import aiohttp
import numpy as np

async def benchmark(url: str, framework: str, qps: int = 100, duration: int = 300):
    """Drive sustained QPS and collect p50/p99 latency."""
    latencies = []
    interval = 1.0 / qps
    
    # Sample 512-token input (randomized for cache fairness)
    np.random.seed(42)
    token_ids = np.random.randint(0, 30522, size=(1, 512)).tolist()
    
    def make_payload(framework: str):
        if framework == "torchserve":
            return json.dumps({"input": token_ids})
        elif framework == "triton":
            return json.dumps({
                "inputs": [{
                    "name": "input_ids",
                    "shape": [1, 512],
                    "datatype": "INT64",
                    "data": token_ids
                }]
            })
        elif framework == "rayserve":
            return json.dumps({"input_ids": token_ids})
        else:
            return json.dumps({"input_ids": token_ids})
    
    payload = make_payload(framework)
    headers = {"Content-Type": "application/json"}
    
    async with aiohttp.ClientSession() as session:
        start_time = time.monotonic()
        tasks = []
        
        async def single_request():
            t0 = time.monotonic()
            try:
                async with session.post(url, data=payload, headers=headers) as resp:
                    await resp.read()
                    lat_ms = (time.monotonic() - t0) * 1000
                    latencies.append(lat_ms)
            except Exception as e:
                print(f"Request failed: {e}")
        
        # Generate requests at target QPS
        while time.monotonic() - start_time < duration:
            tasks.append(asyncio.create_task(single_request()))
            await asyncio.sleep(interval)
        
        await asyncio.gather(*tasks)
    
    sorted_lats = sorted(latencies)
    p50 = statistics.median(sorted_lats)
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)]
    throughput = len(latencies) / duration
    
    print(f"Framework: {framework}")
    print(f"Requests completed: {len(latencies)}")
    print(f"Throughput: {throughput:.1f} req/s")
    print(f"p50 latency: {p50:.1f} ms")
    print(f"p99 latency: {p99:.1f} ms")
    
    return {"p50": p50, "p99": p99, "throughput": throughput}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--framework", required=True,
                        choices=["torchserve", "triton", "rayserve", "ort"])
    parser.add_argument("--qps", type=int, default=100)
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()
    
    asyncio.run(benchmark(args.url, args.framework, args.qps, args.duration))
```

## Hidden costs every framework documentation buries

Every serving framework has costs that do not appear in the getting-started guide. These are the costs that surface in production, usually at the worst possible moment.

![Hidden startup and operational costs per framework](/imgs/blogs/choosing-your-serving-stack-7.png)

### TorchServe: the JVM memory tax

TorchServe runs a Java process as its server component. Before a single model is loaded, TorchServe's JVM allocates approximately 400 MB of heap for the server infrastructure. With 4 worker processes (the default for a T4 instance), each worker adds another 150–200 MB of RAM for its Python interpreter and PyTorch runtime. Total baseline memory on a t3.xlarge (16 GB RAM) before any model weights: approximately 1.2–1.6 GB. On a machine serving multiple models or co-located with other services, this is significant.

The JVM also introduces a cold start latency: the first request after a restart takes 2–5 seconds as the JVM initializes and the model archive is deserialized. This is rarely visible in benchmark runs that include a warmup period, but it is critical for scale-to-zero deployments where cold starts are frequent.

The fix: set `vmargs=-Xmx512m` in `config.properties` to cap JVM heap, and pre-load models at startup with `preload_model=true` to avoid cold starts in production.

### Triton IS: the config.pbtxt learning curve

Triton IS's model repository format requires a `config.pbtxt` (a Protobuf text format file) that specifies input/output tensor names, shapes, data types, batch size, instance group, and dynamic batching configuration. This config file is not validated until the model is loaded, and error messages when it fails are often cryptic.

The time-to-first-working-Triton-endpoint is approximately 30 minutes for an engineer who knows ONNX and is reasonably comfortable with ML infrastructure. For an engineer encountering Triton for the first time, expect 2–4 hours including debugging the inevitable shape mismatch between the ONNX graph's input names and what `config.pbtxt` specifies.

The fix: start with Triton's `--model-control-mode=explicit` and `--strict-model-config=false`. In `strict=false` mode, Triton auto-detects input/output shapes from the ONNX graph, generating a reasonable `config.pbtxt` that you can then copy and modify. This cuts first-model setup time to about 10 minutes.

A sample minimal `config.pbtxt` for BERT-base ONNX:

```protobuf
name: "bert"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [512]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [512]
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    dims: [512]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [2]
  }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```

### Ray Serve: actor overhead at low QPS

Ray Serve's actor model has overhead that matters most at low QPS. Each Ray actor is a separate Python process with its own memory space and event loop. Creating an actor takes 50–200 ms (the cold start time for a new Python process plus model loading). At 1,000+ QPS, this overhead is amortized across enough requests to be negligible. At 5–10 QPS — typical during off-peak hours or early in a product's lifecycle — the overhead is proportionally significant.

More concretely: a Ray Serve deployment with `min_replicas=1` and `max_replicas=10` will spin up additional actors as QPS increases. Each new actor takes 200–500 ms to become ready (process spawn + model load). During this warmup window, requests may queue or be routed to the single warm actor, causing p99 spikes. This is expected Ray Serve behavior documented in the autoscaling guide — but teams migrating from TorchServe's always-warm worker pool are often surprised by it.

The fix: set `init_replicas=2` for any endpoint where p99 spikes during scale-out are unacceptable, and tune `autoscaling_config.upscale_delay_s` to match your actual ramp-up rate.

### vLLM: the full GPU memory pre-allocation

vLLM pre-allocates GPU memory for its KV cache at startup using the `gpu_memory_utilization` parameter (default: 0.9). On an A100 80GB GPU loading Llama-3-8B at FP16 (approximately 16 GB of weights), vLLM will allocate an additional 56 GB for the KV cache (90% of 80GB minus 16GB weights). The GPU shows 72 GB used immediately on startup, before any request arrives.

This is intentional. Pre-allocation eliminates fragmentation and allocation overhead during inference. But it means you cannot co-locate a vLLM process with other GPU workloads on the same GPU, and you cannot run multiple vLLM processes on the same GPU without careful `gpu_memory_utilization` tuning.

```bash
# Limit vLLM KV cache to 70% of GPU memory (leaves ~24GB for co-located workloads)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --gpu-memory-utilization 0.70 \
    --max-model-len 4096 \
    --port 8000
```

The memory math for Llama-3-8B on A100 80GB:
- Model weights (FP16): $2 \times 8\text{B} = 16\text{ GB}$
- KV cache with `gpu_memory_utilization=0.9`: $(80 - 16) \times 0.9 \approx 57.6\text{ GB}$
- KV cache per token per layer (Llama-3-8B, FP16): $2 \times 2 \times 4096 \times 128 \times 32 = 67.1\text{ MB per token slot}$
- Maximum concurrent token capacity: $\lfloor 57.6\text{ GB} / 67.1\text{ MB} \rfloor \approx 858\text{ token slots}$

This is why vLLM's `--max-num-seqs` (default 256) and `--max-model-len` parameters matter: they determine how many concurrent requests can be in flight and how long their context can be, which together bound the KV cache footprint.

## QPS-tier to framework mapping

![QPS tier fitness across serving frameworks](/imgs/blogs/choosing-your-serving-stack-5.png)

The QPS-tier matrix is the second filter after workload type. Read it as: at your expected sustained QPS, which frameworks have no known scaling ceiling within your range?

The key insight from this matrix is that the "Fails SLO" cells at 1,000+ QPS for TorchServe and BentoML are not hypothetical — they reflect real failure modes observed in production. TorchServe's Python worker pool saturates because the JVM management layer becomes a bottleneck at very high request rates. BentoML's Python-first serving engine lacks the C++ acceleration that Triton and vLLM's schedulers provide.

Teams often discover this ceiling in the wrong order: they build a system that works fine at development-scale QPS, present it to product as production-ready, and then discover the ceiling during a load test a week before launch. The fix at that point — migrating to Triton IS or restructuring for Ray Serve autoscaling — is a multi-week project.

The preventive measure is a simple back-of-envelope calculation at architecture time:

$$\text{Max QPS per replica} = \frac{1000}{\text{p99 latency (ms)}} \times \text{GPU utilization}^{-1}$$

For TorchServe eager serving BERT-base at 47 ms p99 and 38 percent GPU utilization:

$$\text{Max QPS} = \frac{1000}{47} \times (0.38)^{-1} \approx 21 \times 2.6 \approx 56 \text{ req/s per replica}$$

To sustain 1,000 QPS you would need approximately 18 TorchServe replicas. On a fleet where each T4 GPU costs \$0.35/hour (spot pricing), 18 replicas = \$75/day just for this one model. Triton IS with TRT achieves 127 req/s per T4 (the benchmark result above), requiring only 8 replicas for the same QPS, at \$33/day. The framework choice is a direct cost multiplier.

#### Worked example: sizing a BERT text classification service

Your team is deploying a BERT-base-uncased sentiment classifier for a product review moderation pipeline. The product team expects 500 QPS peak, with 5-minute burst peaks to 800 QPS. Your SLA is p99 < 50 ms. Hardware budget is 4x T4 GPUs on a g4dn.12xlarge (\$3.91/hour on-demand).

**Option A: TorchServe eager**

From the benchmark: 56 effective req/s per T4 at p99 < 50 ms. 4 T4s = ~224 req/s sustained capacity. Peak QPS 800 / 224 = 3.6x oversubscription — the service will start queuing at 224 QPS and SLA will collapse well before 500 QPS.

**Option B: Triton IS with TRT FP16**

From the benchmark: ~127 req/s per T4 at p99 = 11 ms. 4 T4s = ~508 req/s sustained capacity. Peak QPS 800 / 508 = 1.57x oversubscription — still over-subscribed at 5-minute bursts, but autoscaling HPA can add a g4dn.4xlarge (\$1.25/hour) with 1 additional T4 for burst capacity. Total cost: \$3.91 + burst \$1.25 = \$5.16/hour for a fully SLA-compliant service.

**Option C: TorchServe TorchScript**

From the benchmark: ~108 req/s per T4 at p99 = 31 ms. 4 T4s = ~432 req/s. Still insufficient for 500 QPS sustained. Would require a 5th T4, which is not available on g4dn.12xlarge.

**Conclusion**: Triton IS with TRT FP16 is the right choice. The 35-minute one-time setup cost to build the TRT engine and write `config.pbtxt` is paid once, and the framework-level efficiency gain allows the entire service to run on hardware that is already provisioned.

## Migration paths

Choosing the wrong framework initially is not a catastrophe, provided you know the migration path. Here are the three most common migrations and the specific steps for each.

![Migration timeline: TorchServe to Triton IS](/imgs/blogs/choosing-your-serving-stack-6.png)

### Migration A: TorchServe to Triton IS (add TRT backend)

**When to migrate**: your model has reached 300+ QPS, GPU utilization is below 60 percent on TorchServe, and Triton IS with TRT shows 3x+ improvement in the benchmark above.

**Step 1: Export to ONNX**

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

dummy_input = tokenizer(
    "benchmark text for export",
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True
)

# Export with dynamic batch axis
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"],
     dummy_input["token_type_ids"]),
    "bert.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "token_type_ids": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17
)
print("ONNX export complete — validate accuracy before TRT build")
```

**Step 2: Validate ONNX accuracy** against your gold dataset. Acceptable tolerance: accuracy difference < 0.1% absolute. If this fails, check that your model has no custom ops unsupported by ONNX opset 17.

**Step 3: Build TRT engine** (requires NVIDIA GPU of the same architecture as production):

```bash
# Build TRT FP16 engine using polygraphy (part of TensorRT package)
polygraphy convert bert.onnx \
    --backend trt \
    --fp16 \
    --trt-min-shapes input_ids:[1,512] attention_mask:[1,512] token_type_ids:[1,512] \
    --trt-opt-shapes input_ids:[16,512] attention_mask:[16,512] token_type_ids:[16,512] \
    --trt-max-shapes input_ids:[32,512] attention_mask:[32,512] token_type_ids:[32,512] \
    --save-engine bert_fp16.plan \
    --verbose
```

**Step 4: Set up Triton model repository** with the `config.pbtxt` from the earlier example, placing `bert_fp16.plan` as `1/model.plan`.

**Step 5: Shadow mode validation** — run both TorchServe and Triton side by side, routing 5% of traffic to Triton and comparing outputs. Once accuracy matches and latency is confirmed, shift to 100%.

### Migration B: Ray Serve to vLLM (LLM outgrows Python serving)

**When to migrate**: your Llama-3 or similar model is hitting the throughput ceiling of Ray Serve's Python serving loop, typically visible as GPU utilization below 60 percent at sustained QPS or TTFT growing beyond 1 second at 20+ concurrent users.

![Ray Serve to vLLM: LLM throughput transformation](/imgs/blogs/choosing-your-serving-stack-8.png)

```python
# BEFORE: Ray Serve with transformers (hits ceiling ~120 tok/s)
from ray import serve
from transformers import pipeline
import torch

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1}
)
class LlamaServe:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3-8B-Instruct",
            device="cuda",
            torch_dtype=torch.float16
        )
    
    async def __call__(self, request):
        data = await request.json()
        result = self.pipe(
            data["prompt"],
            max_new_tokens=data.get("max_new_tokens", 256)
        )
        return {"text": result[0]["generated_text"]}

serve.run(LlamaServe.bind())
```

```python
# AFTER: vLLM with OpenAI-compatible API (720 tok/s on A100)
# Start the vLLM server:
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3-8B-Instruct \
#     --gpu-memory-utilization 0.90 \
#     --max-model-len 8192 \
#     --tensor-parallel-size 1 \
#     --port 8000

# Client code (OpenAI-compatible, unchanged from Ray Serve client perspective):
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="placeholder")

async def generate(prompt: str, max_tokens: int = 256) -> str:
    response = await client.chat.completions.create(
        model="meta-llama/Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False
    )
    return response.choices[0].message.content
```

The migration is primarily a server-side change. If your clients already use an OpenAI-compatible REST API format, the client code is unchanged. vLLM's `entrypoints.openai.api_server` speaks the OpenAI Chat Completions API natively.

### Migration C: BentoML to KServe (when you need Kubernetes-native serving)

**When to migrate**: your team has moved from a simple Docker Compose deployment to Kubernetes, and you need HPA, pod disruption budgets, Kubernetes RBAC, and GitOps-compatible serving manifests.

BentoML integrates with KServe via its `bentoml.io` serialization format. The migration involves building your Bento as usual, pushing it to a container registry, and then creating a KServe `InferenceService` manifest that references it:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: bert-classifier
  namespace: ml-serving
  annotations:
    autoscaling.knative.dev/target: "100"  # scale-to-100-concurrent-requests
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 10
    containers:
      - name: kserve-container
        image: your-registry/bert-bento:v1.2.3
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "8Gi"
          requests:
            nvidia.com/gpu: "1"
            memory: "6Gi"
        env:
          - name: BENTO_PORT
            value: "8080"
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

The key advantage of KServe over a bare Kubernetes Deployment for BentoML services: built-in support for canary rollouts via the `InferenceService` traffic split spec, integration with KNative for scale-to-zero, and standardized prediction logging to cloud storage.

## Multi-framework serving: vLLM and Triton behind Nginx

Real production ML systems often serve multiple model types. A recommendation system might have a BERT-based recall model (CV/encoder, good fit for Triton) and a Llama-3-based reranker (LLM, good fit for vLLM) behind a single API endpoint. Running these on separate servers and routing between them at the Nginx layer is the cleanest architecture.

![Multi-framework serving: vLLM and Triton behind a single Nginx router](/imgs/blogs/choosing-your-serving-stack-4.png)

```nginx
# nginx.conf: multi-framework routing
upstream vllm_backend {
    server vllm-service:8000;
    keepalive 64;
}

upstream triton_backend {
    server triton-service:8000;
    keepalive 64;
}

server {
    listen 80;
    
    # LLM endpoints → vLLM
    location /v1/chat/ {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 120s;
        proxy_buffering off;  # Required for SSE token streaming
    }
    
    location /v1/completions {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 120s;
        proxy_buffering off;
    }
    
    # Embedding / classification → Triton IS
    location /v2/models/ {
        proxy_pass http://triton_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 30s;
    }
    
    # Health check aggregation
    location /health {
        add_header Content-Type application/json;
        return 200 '{"status": "ok", "backends": ["vllm", "triton"]}';
    }
}
```

And the accompanying Docker Compose for local development/staging:

```yaml
version: "3.9"

services:
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - vllm
      - triton
    networks:
      - ml-net

  vllm:
    image: vllm/vllm-openai:latest
    command: >
      --model meta-llama/Llama-3-8B-Instruct
      --gpu-memory-utilization 0.85
      --max-model-len 8192
      --port 8000
      --tensor-parallel-size 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]  # GPU 0 for LLM
              capabilities: [gpu]
    environment:
      - HF_TOKEN=${HF_TOKEN}
    networks:
      - ml-net

  triton:
    image: nvcr.io/nvidia/tritonserver:24.04-py3
    command: >
      tritonserver
      --model-repository=/models
      --http-port=8000
      --grpc-port=8001
      --model-control-mode=poll
      --repository-poll-secs=30
    volumes:
      - ./model_repository:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]  # GPU 1 for CV
              capabilities: [gpu]
    networks:
      - ml-net

networks:
  ml-net:
    driver: bridge
```

This architecture gives you LLM requests routed to vLLM (with SSE token streaming, OpenAI-compatible API, PagedAttention) and CV/encoder requests routed to Triton IS (with TRT backend, dynamic batching, ensemble pipelines) — all from the same external API surface.

The key Nginx configuration detail: `proxy_buffering off` on the vLLM location. Without this, Nginx buffers the entire response before forwarding it to the client, which defeats the purpose of SSE streaming. Token streaming requires both `proxy_buffering off` and a `proxy_read_timeout` that exceeds your maximum generation time.

## Case studies: framework choices at real scale

### Case study 1: Triton IS at NVIDIA's inference reference benchmarks

NVIDIA's MLPerf Inference benchmarks run Triton IS as the reference implementation for several workloads. In MLPerf Inference v4.0 (2024), Triton IS with TRT FP8 on H100 SXM5 achieves 12,890 samples/second for ResNet-50 offline inference and 2,247 samples/second for BERT-large. These are the highest published throughput numbers for these models on publicly available hardware. The combination of TensorRT FP8 precision (H100 native), multi-instance GPU (MIG) support, and Triton's dynamic batching scheduler is what produces these numbers. They represent the hard ceiling of what is achievable with current hardware — and Triton IS is the serving layer in front of all of it.

### Case study 2: vLLM at Anyscale

Anyscale's 2023 vLLM benchmarks (the paper that introduced PagedAttention, Kwon et al. 2023) showed that vLLM achieves 3–24x higher throughput than HuggingFace Transformers and up to 3.5x higher throughput than FasterTransformer on OPT-13B and LLaMA-7B, depending on request concurrency. At 50 concurrent requests, vLLM's PagedAttention eliminated the memory waste of static KV allocation, allowing the KV cache to fit 12.3x more requests in the same GPU memory. The throughput improvement at high concurrency scales directly with KV cache utilization.

This is not a laboratory result — Anyscale runs vLLM in production serving commercial LLM API traffic. The claim that "vLLM is the right choice for LLM serving" is backed by both published benchmarks and production deployment at scale.

### Case study 3: Ray Serve at Shopify

Shopify's ML platform team published in 2023 that they use Ray Serve for online prediction serving across their recommendation, fraud detection, and personalization models. The key advantage cited: Ray Serve's Python-native deployment model allowed their ML team to iterate on model logic without involving infrastructure engineers. Ray Serve handles the scaling, routing, and health checking. The ML team handles the model code. This separation of concerns is a genuine operational advantage for teams where ML engineers outnumber ML infrastructure engineers.

The throughput numbers in Shopify's deployment are not public, but they report sustaining p99 latency < 50 ms for their highest-traffic prediction endpoints. This validates Ray Serve's fitness at the 100–1000 QPS tier for Python-native ML workloads.

### Case study 4: TorchServe in AWS SageMaker

AWS SageMaker's multi-model server is built on TorchServe. When you deploy a PyTorch model via SageMaker's `create_model` API with a `.tar.gz` containing `model.pth` and a `torchserve` handler, SageMaker is running TorchServe as the inference process. This is the most widely deployed TorchServe installation in the industry, serving millions of inference requests per day across SageMaker customers.

The implication: if your team is using SageMaker for model deployment, you are already using TorchServe. Understanding TorchServe's handler API, its management API endpoints, and its batching configuration is directly applicable to your SageMaker deployment. The SageMaker wrapper does not change the underlying serving behavior.

## Operational comparison: monitoring, debugging, and day-2 operations

Framework selection is not just an architecture decision — it is an on-call burden allocation decision. The framework you choose determines what breaks at 3 AM and how hard it is to diagnose. Here is what day-2 operations look like across the major frameworks.

### Observability surfaces

**TorchServe** emits Prometheus-compatible metrics at `:8082/metrics` including `ts_inference_latency_microseconds` (histogram), `ts_queue_latency_microseconds` (histogram), `ts_inference_requests_total` (counter), and GPU utilization via `nvidia.com/gpu_utilization` when the metrics collector is configured. These are usable in a standard Prometheus + Grafana setup without custom instrumentation. Example Prometheus alert for p99 latency:

```yaml
# prometheus/rules/torchserve.yml
groups:
  - name: torchserve_slo
    rules:
      - alert: TorchServeP99LatencyHigh
        expr: |
          histogram_quantile(0.99,
            rate(ts_inference_latency_microseconds_bucket[5m])
          ) / 1e6 > 0.050
        for: 2m
        labels:
          severity: page
        annotations:
          summary: "TorchServe p99 latency > 50ms"
          description: "Model {{ $labels.model_name }} p99 = {{ $value | humanizeDuration }}"
```

**Triton IS** emits extensive metrics at `:8002/metrics` including per-model request counts, latency histograms, queue time histograms, GPU memory usage, and compute utilization. The metric granularity is the best in the ecosystem: `nv_inference_request_duration_us` is a histogram broken down by model name and version. Triton's detailed queue time metric is particularly valuable for diagnosing batching configuration issues — if queue time is high relative to compute time, your `max_queue_delay_microseconds` needs tuning.

**Ray Serve** emits metrics via Ray's built-in Prometheus integration at the Ray dashboard port. `serve_num_ongoing_requests` (gauge per deployment), `serve_backend_request_batch_size` (histogram), and `serve_backend_queued_queries` are the key signals. The Ray dashboard at `:8265` provides a visual view of replica health and request throughput. Debugging Ray Serve issues often starts at the actor level: `ray list actors` and checking actor logs for exceptions.

**vLLM** emits metrics at `/metrics` including `vllm:num_requests_running` (active sequences in continuous batch), `vllm:num_requests_waiting` (queue depth), `vllm:gpu_cache_usage_perc` (KV cache utilization), and `vllm:generation_tokens_total`. The KV cache utilization metric is critical for LLM serving: when it approaches 100 percent, vLLM starts evicting KV cache blocks (preemption), which causes latency spikes. An alert on `vllm:gpu_cache_usage_perc > 0.90` gives you 30–60 seconds of warning before preemption impacts SLOs.

```yaml
# prometheus/rules/vllm.yml
groups:
  - name: vllm_slo
    rules:
      - alert: VLLMKVCacheNearFull
        expr: vllm:gpu_cache_usage_perc > 0.90
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "vLLM KV cache > 90% — preemption imminent"
          
      - alert: VLLMQueueDepthHigh
        expr: vllm:num_requests_waiting > 20
        for: 30s
        labels:
          severity: page
        annotations:
          summary: "vLLM request queue > 20 — scale out or reduce concurrency"
```

### Common failure modes per framework

**TorchServe OOM**: the most common TorchServe production incident is a worker process OOM when batch size is set too high relative to model size. TorchServe does not pre-validate that `batch_size * model_size_gb < GPU_memory_gb`. The symptom is worker processes dying with CUDA OOM errors and TorchServe returning HTTP 503. Mitigation: set `max_batch_size` conservatively at deploy time and increase only after load testing. Monitor `ts_inference_requests_total` with HTTP 503 status code as a leading indicator.

**Triton config mismatch**: the most common Triton production incident is a `config.pbtxt` that specifies the wrong input shape or tensor name, causing all requests to fail with `INVALID_ARGUMENT: unexpected shape for input`. This often happens when the ONNX model is retrained and exported with changed input names (e.g., `input_ids` → `input`). The fix: Triton's model control API allows hot-swapping `config.pbtxt` without a server restart — `POST /v2/repository/models/{model_name}/unload` followed by updating the config file and `POST /v2/repository/models/{model_name}/load`.

**Ray Serve actor death**: when a Ray Serve actor (replica) crashes — due to OOM, an unhandled exception, or a SIGKILL from the OS — Ray detects the actor death and restarts it. During the restart window (typically 1–5 seconds for model loading), the deployment routes to the remaining healthy replicas. If `min_replicas=1` and the single replica dies, requests queue until it restarts. Always set `min_replicas=2` for production deployments where actor restarts are unacceptable.

**vLLM TTFT spike under KV cache pressure**: when vLLM's KV cache fills up, it applies preemption — pausing decode for lower-priority sequences and reallocating their KV blocks to higher-priority ones. The preempted sequences restart their decode from scratch (re-prefill). This causes p99 TTFT spikes because the re-prefilled sequence now has to regenerate all its KV cache. Monitor `vllm:num_preemptions_total` as a counter; if it is incrementing in production, either reduce `max-model-len`, reduce `max-num-seqs`, or add more GPU capacity.

### Framework comparison: operational burden scoring

| Dimension | TorchServe | Triton IS | Ray Serve | vLLM |
|---|---|---|---|---|
| Time to diagnose p99 spike | 15 min | 20 min | 25 min | 10 min |
| Config change without restart | Yes (management API) | Yes (model control API) | Yes (rolling update) | No (requires restart) |
| Multi-GPU support | Manual (separate instances) | First-class (instance group) | Actor-level GPU affinity | `--tensor-parallel-size` |
| Model hot-swap | Yes (`register`, `unregister`) | Yes (poll mode) | Yes (rolling deploy) | No (restart required) |
| Memory leak detection | JVM heap monitoring | `nv_gpu_memory_used_bytes` | Ray Object Store metrics | KV cache utilization |
| On-call learning curve | Low | Medium | Medium | Low (for LLM) |

The time-to-diagnose advantage for vLLM comes from the specificity of its metrics: `vllm:num_requests_waiting` tells you immediately whether the issue is request queuing (scale out) or generation speed (model or hardware issue). TorchServe's coarser metrics require more inference to reach the same diagnosis.

## The "start here" recommendation

If you are a team building your first production serving system — you have one or two models, you want reliable production serving without a six-week infrastructure project, and you are not serving an LLM — the recommendation is clear: **start with TorchServe or ONNX Runtime + FastAPI**.

Here is the precise decision:

- **PyTorch model, want managed server with batching, metrics, management API**: TorchServe. It is battle-tested (AWS SageMaker runs on it), requires minimal configuration for a standard PyTorch model, and handles the operational basics (health checks, multiple workers, metrics) without infrastructure engineering work.

- **Any model framework, want minimal-dependency, cross-platform inference**: ONNX Runtime with the CUDA execution provider wrapped in FastAPI. Export to ONNX once, write 60 lines of FastAPI, get hardware-accelerated inference with a standard REST API. When you need batching, add it to the FastAPI handler. When you need scaling, add a Kubernetes Deployment with HPA. The interface layer stays thin and legible.

The reason to start here rather than jumping straight to Triton IS or Ray Serve: you will learn what your actual bottlenecks are. Teams that start with Triton IS on day one often spend two weeks debugging `config.pbtxt` files before the first successful inference. Teams that start with TorchServe or ORT + FastAPI have a working production system in a day, and when they hit a real bottleneck they know exactly what it is and can migrate to the right framework with evidence.

The migration from TorchServe to Triton IS is well-trodden and documented above. The migration from ORT + FastAPI to Ray Serve is even simpler — it is mostly wrapping the existing FastAPI handler in a Ray Serve `@serve.deployment` decorator and letting Ray handle the scaling.

Do not start with Triton IS unless you have a team member who has operated it before, or unless your benchmark data already shows you need TRT-level throughput from day one. Do not start with vLLM unless you are serving an LLM — it is not a general-purpose serving framework and should not be treated as one.

#### Worked example: first production serving system, NLP startup

Your team is a five-person NLP startup shipping a document classification product. You have one model: a fine-tuned BERT-base classifier. Expected launch traffic: 20–50 QPS. Expected three-month traffic: 100–200 QPS. Infrastructure: one A10G GPU on an on-demand AWS instance (`g5.xlarge`, \$1.006/hour). Team: 3 ML engineers who know PyTorch, 1 infrastructure engineer who knows Docker and Kubernetes basics.

**Recommended choice: TorchServe**

Setup process (total time: ~2 hours):

```bash
# 1. Install TorchServe
pip install torchserve torch-model-archiver torch-workflow-archiver

# 2. Create handler (bert_handler.py — implement preprocess/inference/postprocess)
# See torchserve deep-dive post for handler template

# 3. Package model
torch-model-archiver \
    --model-name bert_classifier \
    --version 1.0 \
    --serialized-file bert_finetuned.pth \
    --handler bert_handler.py \
    --extra-files "config.json,vocab.txt" \
    --export-path model_store/

# 4. Start server with 2 GPU workers
torchserve \
    --start \
    --model-store model_store/ \
    --models bert_classifier=bert_classifier.mar \
    --ts-config config.properties \
    --foreground
```

`config.properties`:

```properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=200
default_workers_per_model=2
default_response_timeout=60
install_py_dep_per_model=false
vmargs=-Xmx512m -XX:MaxDirectMemorySize=512m
```

At 50 QPS on an A10G: TorchServe with TorchScript will achieve approximately p99 < 25 ms and GPU utilization 45–55 percent. This is well within your launch SLA.

When you hit 150+ QPS at three months, your monitoring will show GPU utilization rising toward 80 percent and p99 latency creeping up. At that point, the Triton IS migration path above takes one sprint to execute. You will have three months of production operational experience, benchmarks from your actual traffic pattern, and a clear picture of your actual bottleneck.

Cost for this system at 100 QPS sustained: `g5.xlarge` on-demand at \$1.006/hour = \$24.14/day. For \$0.002 per inference request at 100 QPS this represents \$17.28/day in inference value on \$24.14/day of compute — a 72 percent gross margin on GPU capacity, before the Triton optimization that would raise it further.

## When to use each framework (and when not to)

**TorchServe — use when**: PyTorch shop, single model, team unfamiliar with C++ config, AWS SageMaker deployment target, daily model updates via management API. **Do not use when**: LLM serving is primary workload, QPS > 500 per replica, multi-framework CV pipelines requiring TRT ensemble.

**Triton IS — use when**: GPU-optimized CV pipeline, TensorRT is required, multi-model serving from a single server, 1,000+ QPS requirement, NVIDIA-native CUDA optimization is needed, team has C++/protobuf comfort. **Do not use when**: team is Python-only, model updates are daily (config changes require careful versioning), first serving system with no Triton experience.

**ONNX Runtime — use when**: cross-platform portability, AMD/Intel/ARM hardware, encoder-only transformers, embedding models, lowest-dependency production serving. **Do not use when**: batching and multi-worker management are needed (ORT has no built-in server — wrap in FastAPI or use Triton as ORT backend), LLM autoregressive decoding is required.

**Ray Serve — use when**: Python-first ML microservice, multi-model composition, complex preprocessing chains, team already uses Ray for training, 100–1,000 QPS range with autoscaling, rapid iteration on model logic. **Do not use when**: LLM serving at high concurrency (GPU GIL stalls), team lacks Ray operational experience, cold start latency is unacceptable (actor spin-up time).

**BentoML — use when**: model packaging and reproducible deployment artifacts, handing off models to platform teams, prototyping with a clean deployment path to production, first serving system where simplicity matters most. **Do not use when**: QPS > 200 per replica, LLM serving, maximum performance is required.

**vLLM — use when**: LLM inference (Llama, Mistral, Qwen, Falcon, any HuggingFace autoregressive transformer), 20+ concurrent users, PagedAttention is required, OpenAI-compatible API is desired. **Do not use when**: serving encoder-only models (BERT, RoBERTa, sentence-transformers) where it provides no advantage, GPU memory is shared between workloads (100% pre-allocation is problematic), team is not yet serving an LLM.

**TGI — use when**: HuggingFace-native ecosystem, deep transformers integration, token streaming is a hard requirement, multi-GPU tensor parallelism via `--num-shard` is needed. **Do not use when**: model is not on HuggingFace Hub, you need a framework that is model-source-agnostic.

## Key takeaways

1. **Model type is the highest-signal selection axis**: LLM → vLLM or TGI; CV TRT pipeline → Triton IS; Python ML microservice → Ray Serve; single PyTorch model → TorchServe.

2. **QPS tier is the second filter**: below 100 QPS, any framework works; above 1,000 QPS, only Triton IS and vLLM/TGI have been demonstrated at this scale without massive over-provisioning.

3. **The four hidden costs to budget before choosing**: TorchServe JVM 400 MB baseline, Triton IS 30-minute `config.pbtxt` learning curve, Ray Serve actor 50–200 ms cold start, vLLM 100% GPU memory pre-allocation at startup.

4. **Triton IS with TRT FP16 delivers 4x lower p99 latency than TorchServe eager at 100 QPS** on T4 hardware (11 ms vs 47 ms). This is the cost-reduction multiplier to calculate when deciding whether the setup investment is worth it.

5. **Multi-framework serving behind Nginx is a legitimate production pattern**: vLLM for LLM traffic, Triton IS for CV traffic, a shared external API surface. The Nginx `proxy_buffering off` flag is required for SSE token streaming.

6. **The migration paths are well-trodden**: TorchServe → Triton IS (ONNX export + TRT build + config.pbtxt = 1–2 weeks), Ray Serve → vLLM (drop-in OpenAI-compatible API = 1–3 days), BentoML → KServe (container registry + InferenceService YAML = 1 week).

7. **Start simple, migrate with evidence**: begin with TorchServe or ORT + FastAPI, establish a production baseline, then migrate when benchmarks show a clear bottleneck. Premature framework optimization is as costly as premature code optimization.

8. **Framework choice is a cost multiplier**: the difference between TorchServe eager and Triton TRT at 1,000 QPS is approximately 18 replicas vs 8 replicas on T4 GPUs — a 2.25x difference in infrastructure cost for the same throughput. Framework selection is infrastructure budget allocation.

9. **vLLM is not a general-purpose serving framework**: its architecture (PagedAttention, continuous batching) is purpose-built for autoregressive LLM inference. Using it for BERT classification or XGBoost serving provides no advantage and adds unnecessary complexity.

10. **Python flexibility and GPU optimization depth are inversely correlated across the framework landscape**: the frameworks with the deepest GPU optimization (Triton IS with TRT, vLLM) have the most constrained Python flexibility. The frameworks with the most Python flexibility (Ray Serve, BentoML) are furthest from hardware-optimal inference.

## Putting it together: the one-page decision guide

The selection process compressed into a sequence you can run in under five minutes:

1. **LLM?** Yes → vLLM (or TGI for HuggingFace native). Skip everything else.
2. **CV pipeline with TensorRT?** Yes → Triton IS. Skip steps 3–5.
3. **QPS > 500?** Yes → Triton IS or Ray Serve with autoscaling. Skip steps 4–5.
4. **Team Python-first, complex preprocessing?** Yes → Ray Serve.
5. **Single PyTorch model, want zero ops burden?** → TorchServe.
6. **Need packaging artifact for platform handoff?** → BentoML.
7. **Cross-platform or heterogeneous hardware?** → ONNX Runtime + FastAPI.

Any team that can make it through this seven-step filter without ambiguity has their framework. Teams that feel ambiguity between two frameworks at the same step should run the benchmark script from this post on their actual model and QPS target, and let the numbers decide.

The two metrics that resolve most ambiguities: p99 latency at target QPS on actual hardware, and GPU utilization percentage at sustained load. If GPU utilization is below 60 percent at your target QPS with your current framework, switching to a framework with a better batching scheduler will almost certainly reduce both p99 latency and infrastructure cost. If GPU utilization is above 85 percent, you are capacity-constrained rather than framework-constrained — invest in more GPUs rather than framework migration.

Finally, resist the temptation to build your own serving framework from scratch using raw Python and FastAPI unless your workload is genuinely unique. The frameworks above represent thousands of engineer-years of work on batching, concurrency, memory management, and GPU utilization. The typical in-house FastAPI serving loop runs GPU utilization between 25 and 45 percent at production QPS, while TorchServe runs 38–51 percent, and Triton IS runs 67–91 percent. The gap between a raw FastAPI loop and a production serving framework is real and measurable.

## Further reading

- **vLLM Paper**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. The original PagedAttention paper; directly relevant to understanding why vLLM outperforms naive LLM serving.
- **NVIDIA Triton IS documentation**: `docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/`. The canonical reference for `config.pbtxt` syntax, ensemble backend, and performance tuning.
- **Ray Serve documentation**: `docs.ray.io/en/latest/serve/`. The deployment API, autoscaling configuration, and deployment graph composition model.
- **TorchServe documentation**: `pytorch.org/serve/`. Handler API reference, management API, metrics configuration, and the `.mar` archive format.
- **Little's Law and Queueing Theory for Engineers**: Cooper (1981), "Introduction to Queueing Theory." The mathematical foundation for the SLO-aware capacity planning models in this post.
- [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — series introduction, the SLO triangle, and the latency/throughput/cost framework that underpins all framework choices.
- [Triton Inference Server deep dive](/blog/machine-learning/model-serving/triton-inference-server-deep-dive) — detailed coverage of `config.pbtxt`, ensemble pipelines, and dynamic batching configuration.
- [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) — chunked prefill, prefix caching, and speculative decoding internals that explain vLLM's throughput advantage.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — series capstone: full decision tree from model type to production architecture, including the multi-framework patterns introduced in this post.
