---
title: "BentoML and MLServer: two philosophies for packaging ML services"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master BentoML's Python-first service definition and MLServer's V2 inference protocol to package, containerize, and serve any ML model with the right tool for your team's stack."
tags:
  [
    "model-serving",
    "inference",
    "bentoml",
    "mlserver",
    "kserve",
    "v2-inference-protocol",
    "ml-infrastructure",
    "seldon",
    "containerization",
    "adaptive-batching",
    "model-packaging",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/bentoml-and-mlserver-1.png"
---

It is Thursday morning and your team has a scikit-learn classifier that has been producing excellent offline metrics for two weeks. Someone opens a ticket: "deploy it before Friday." You have a few hours. If your team owns a Kubernetes cluster with KServe already installed, the right move is to drop a twenty-line JSON config file and be done in under an hour. If your team runs a handful of VM-based microservices and nobody has touched Kubernetes, the right move is to write a BentoML service class, run `bentoml build`, and ship a Docker image that installs the same way as every other service in your stack.

The two tools covered in this post — BentoML and MLServer from Seldon — make the same fundamental promise: take a trained model and turn it into a production HTTP service. But they make that promise through completely different philosophies, and choosing the wrong one for your context costs you days of debugging unfamiliar abstractions. BentoML is Python-first and developer-local: you define your service in Python, build a self-contained OCI image, and run it anywhere Docker runs. MLServer is protocol-first and Kubernetes-native: it implements the Open Inference Protocol (V2), which means your models become interchangeable plug-ins in a KServe or Seldon Core pipeline with standardized typed tensor APIs.

By the end of this post you will understand the complete BentoML lifecycle from `save_model` to a running container, how adaptive batching works and when it earns its cost, the V2 inference protocol wire format and why standardization matters at scale, how MLServer's runtime registry turns a JSON config into a running service, and how KServe orchestrates MLServer pods with traffic splits and health probes. The running example throughout is a scikit-learn RandomForestClassifier trained on the UCI Adult dataset — simple enough to focus on the serving mechanics, real enough to produce meaningful numbers.

![BentoML service lifecycle: from save_model through service definition, bentoml build, containerize, and finally running in production](/imgs/blogs/bentoml-and-mlserver-1.png)

The figure above maps the full BentoML lifecycle. Every stage produces an auditable artifact: a tagged model in the local model store, a Python service class, a Bento directory on disk, and an OCI image. This artifact chain is the core value proposition of BentoML — reproducibility at every stage, not just at inference time.


## Why packaging philosophy matters

The history of ML serving tools has been a story of two competing design philosophies. The first philosophy — call it the application developer's approach — treats model serving as a special case of building a web service. You write code, the framework handles the HTTP layer, and you ship a Docker image. Flask, FastAPI, and BentoML all live in this tradition. The second philosophy — call it the infrastructure operator's approach — treats model serving as a special case of running workloads on a compute cluster. You describe what you want in a manifest file, the platform fulfills it, and you never think about the underlying HTTP layer at all. TensorFlow Serving, Triton Inference Server, MLServer, and KServe live in this tradition.

Neither philosophy is wrong. They are optimized for different problems. The application developer's approach optimizes for iteration speed, custom logic, and debuggability. You can attach a debugger to a BentoML service class the same way you attach one to any other Python process. You can add a `print` statement, rerun, and see the output immediately. The infrastructure operator's approach optimizes for scale, standardization, and operational uniformity. You can describe 100 models in 100 YAML files and deploy all of them with a single `kubectl apply -f models/`. You can write one Prometheus dashboard that covers all 100.

The key insight — and the one that most teams miss when they first encounter both tools — is that the right choice depends almost entirely on **where your team's expertise lives and what your organization's infrastructure looks like**. A team of five data scientists who all know Python will be faster with BentoML even if MLServer is theoretically the better tool for their use case. A platform engineering team running 50 Kubernetes-native microservices will be faster with MLServer even if BentoML has a nicer API.

This post covers both tools in enough depth that you can make an informed decision and execute competently on either path.


## The serving SLO triangle for non-LLM models

Before diving into either tool, it is worth anchoring to the series-wide framework. The SLO triangle — latency, throughput, cost — applies equally to a 50KB scikit-learn model and to a 70B LLM, but the numbers and the bottlenecks are completely different.

For a scikit-learn RandomForestClassifier predicting income bracket, a single inference takes roughly 0.3–2ms of CPU time depending on the number of trees and features. The bottleneck is almost never the model itself. The bottleneck is the serving overhead: Python interpreter startup, HTTP parsing, JSON serialization, framework initialization. A naively wrapped Flask app can serve this model at roughly 80–150 requests per second on a single CPU core. A well-tuned BentoML service with adaptive batching on the same hardware serves 400–600 requests per second. That is a 3–4x throughput difference from packaging decisions alone, with no changes to the model.

For MLServer, the same model behind the V2 protocol on a single CPU core serves 350–500 requests per second using gRPC (because gRPC avoids the JSON serialization overhead that dominates HTTP/JSON serving for tabular data). The V2 protocol's typed tensor encoding means the framework can skip Python dict construction for inputs and outputs entirely, operating on raw binary buffers.

This is why packaging and protocol choices matter for classical ML, not just LLMs. At \$0.10 per CPU core-hour, the difference between 150 req/s and 500 req/s means you need 3.3x fewer cores for the same traffic load — a cost difference that compounds at scale. For a service handling 10,000 requests per second, that is the difference between 67 cores and 20 cores. At AWS on-demand pricing, roughly \$160/day versus \$48/day. Packaging is not bureaucracy. It is the last mile of optimization.


## BentoML: the model store and `@bentoml.service`

BentoML's architecture rests on two independent components that compose: a **model store** that versions and stores trained model artifacts, and a **service class** that defines how those artifacts are exposed as HTTP or gRPC endpoints. Understanding each separately is the key to using BentoML correctly.

### The model store

The model store is a local artifact registry managed by BentoML. It lives at `~/bentoml/models/` by default and is shared across all BentoML projects on your machine. Every saved model receives a unique tag in the format `name:version_hash`, and BentoML tracks metadata including the saving time, the framework (pytorch, sklearn, transformers, onnx), and any custom labels you attach.

```python
import bentoml
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

# Train the model
adult = fetch_openml("adult", version=2, as_frame=True)
X, y = adult.data, adult.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Save to BentoML model store
model_ref = bentoml.sklearn.save_model(
    "adult_income_clf",
    clf,
    signatures={
        "predict": {"batchable": True, "batch_dim": 0},
        "predict_proba": {"batchable": True, "batch_dim": 0},
    },
    labels={"team": "ml-platform", "task": "income-classification"},
    metadata={
        "accuracy": float(clf.score(X_test, y_test)),
        "n_estimators": 200,
        "training_data": "uci_adult_v2",
    },
)

print(f"Model saved: {model_ref.tag}")
# Output: Model saved: adult_income_clf:a1b2c3d4e5f6a7b8
```

The `signatures` dict is important: it tells BentoML which methods on the model object can be called, and whether each is batchable. When `batchable=True` with `batch_dim=0`, BentoML knows that multiple inputs can be stacked along dimension 0 and run through `predict` together — the foundation of adaptive batching.

Other framework-specific save functions follow the same API: `bentoml.pytorch.save_model`, `bentoml.transformers.save_model`, `bentoml.onnx.save_model`, `bentoml.xgboost.save_model`. Each knows how to serialize and deserialize its respective framework's native format. PyTorch models are saved with `torch.save` internally. Transformers models use `save_pretrained`. ONNX models store the `.onnx` protobuf.

```bash
# Inspect the model store
bentoml models list
# NAME                  VERSION   MODULE         CREATED
# adult_income_clf      a1b2c3d4  bentoml_sklearn 2026-06-22 09:14:23

bentoml models get adult_income_clf:latest
# model_name: adult_income_clf
# version: a1b2c3d4e5f6a7b8
# framework: sklearn
# size: 4.2 MB
# labels: {'team': 'ml-platform', 'task': 'income-classification'}
```

### The `@bentoml.service` class: BentoML 1.3+ pattern

BentoML underwent a significant API redesign in version 1.0 (released 2023) that replaced the older "Runner" pattern with a direct service class approach. The Runner pattern required creating a runner object separately, attaching it to a service, and coordinating between them — a model that worked but felt indirect. In BentoML 1.3+, the service class is the primary building block: one Python class defines everything.

```python
# service.py
import bentoml
import numpy as np
from pydantic import BaseModel
from typing import List, Annotated
import pandas as pd

# Input schema using Pydantic v2
class AdultInput(BaseModel):
    age: float
    workclass: str
    fnlwgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

class PredictionOutput(BaseModel):
    income_class: str           # ">50K" or "<=50K"
    confidence: float           # probability of predicted class

@bentoml.service(
    resources={
        "cpu": "2",
        "memory": "2Gi",
    },
    traffic={
        "timeout": 30,
        "max_concurrency": 100,
    },
)
class IncomeClassifier:
    # Load model at startup — runs once per worker process
    bento_model = bentoml.models.get("adult_income_clf:latest")

    def __init__(self):
        self.clf = self.bento_model.load_model()
        # Build column order from training data
        self.feature_cols = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country"
        ]

    @bentoml.api(
        batchable=True,
        max_batch_size=64,
        max_latency_ms=50,
    )
    def predict(self, inputs: List[AdultInput]) -> List[PredictionOutput]:
        """Predict income class for a list of inputs."""
        # Convert Pydantic models to DataFrame
        df = pd.DataFrame([inp.model_dump() for inp in inputs])
        df = df[self.feature_cols]

        classes = self.clf.predict(df)
        probas = self.clf.predict_proba(df)

        results = []
        for cls, proba_row in zip(classes, probas):
            class_idx = list(self.clf.classes_).index(cls)
            results.append(PredictionOutput(
                income_class=cls,
                confidence=float(proba_row[class_idx])
            ))
        return results
```

The `@bentoml.service` decorator at the class level controls resource allocation and traffic shaping. `resources={"cpu": "2", "memory": "2Gi"}` sets the Kubernetes resource request and limit when the bento is deployed to a cluster. `traffic={"timeout": 30, "max_concurrency": 100}` controls the internal HTTP server: any request that does not receive a response in 30 seconds is cancelled, and at most 100 requests can be in-flight simultaneously per worker.

The `@bentoml.api` decorator on each method defines one HTTP endpoint. `batchable=True` enables BentoML's adaptive batching subsystem for that method. `max_batch_size=64` caps the batch at 64 inputs, regardless of how many requests are waiting. `max_latency_ms=50` sets the maximum time the batching window stays open: if fewer than 64 requests have arrived but 50ms have elapsed, the batch fires regardless.

### The runner pattern (deprecated, but you will see it)

If you work with codebases that used BentoML 0.13.x, you will encounter the older Runner pattern. Understanding it helps you read existing code and migrate it.

```python
# The OLD BentoML 0.13 runner pattern — do not write new code this way
import bentoml
from bentoml.io import NumpyNdarray

# Create a runner (a subprocess that owns the model)
runner = bentoml.sklearn.get("adult_income_clf:latest").to_runner()

# Create the service and attach the runner
svc = bentoml.Service("income_classifier", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_data):
    return runner.predict.run(input_data)
```

The Runner pattern spawned a separate process for model inference and the service was a thin coordinator that routed requests to it. This gave isolation and was useful when you needed multiple models orchestrated together. In BentoML 1.x, `@bentoml.depends` replaces this pattern for multi-service composition.

### `@bentoml.depends`: service composition

When you need to chain services — for example, a preprocessing service, a model service, and a postprocessing service — BentoML 1.3 uses `@bentoml.depends`:

```python
@bentoml.service(resources={"cpu": "1", "memory": "512Mi"})
class Preprocessor:
    @bentoml.api
    def transform(self, raw_input: dict) -> np.ndarray:
        # feature engineering logic here
        ...

@bentoml.service(resources={"cpu": "2", "memory": "2Gi"})
class IncomeClassifierWithPreprocessing:
    preprocessor = bentoml.depends(Preprocessor)

    def __init__(self):
        self.clf = bentoml.models.get("adult_income_clf:latest").load_model()

    @bentoml.api
    async def predict(self, raw_input: dict) -> PredictionOutput:
        # Call the dependent service asynchronously
        features = await self.preprocessor.transform(raw_input)
        pred = self.clf.predict(features.reshape(1, -1))
        proba = self.clf.predict_proba(features.reshape(1, -1))
        return PredictionOutput(income_class=pred[0], confidence=float(proba.max()))
```

When you run `bentoml serve service:IncomeClassifierWithPreprocessing --production`, BentoML starts both services and wires them together. Each service can have its own resource allocation, its own replicas, and its own batching configuration. The dependency relationship is explicit in the code rather than in a separate configuration file, which is a significant ergonomic win over Kubernetes-native composition patterns.

![BentoML 1.3+ service architecture: from HTTP clients through the service decorator to the model store and OCI container runtime](/imgs/blogs/bentoml-and-mlserver-2.png)


## `bentofile.yaml` and `bentoml build`

Once your service class is correct, `bentoml build` packages everything into a **Bento** — BentoML's deployment artifact. The Bento is a directory that contains your code, your model artifacts (copied from the model store), your Python dependencies, and Docker build configuration. A `bentofile.yaml` in your project root controls exactly what goes into the Bento.

```yaml
# bentofile.yaml
service: "service:IncomeClassifier"     # module:class

include:
  - "*.py"                              # include all Python files
  - "config/*.yaml"                     # include config files

python:
  packages:
    - scikit-learn>=1.3.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - pydantic>=2.0.0

docker:
  base_image: "python:3.11-slim"
  cuda_version: null                    # no GPU for sklearn
  system_packages:
    - libgomp1                          # OpenMP for sklearn parallel trees
  dockerfile_template: null             # use BentoML's generated Dockerfile
  env:
    NUM_WORKERS: "4"
    BENTOML_LOG_LEVEL: "INFO"
```

For a GPU-based model, the `docker` section changes:

```yaml
docker:
  base_image: "nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04"
  cuda_version: "12.1.0"
  system_packages:
    - libgomp1
    - python3-pip
```

When you run `bentoml build`, the process:

1. Resolves the service class and all `@bentoml.depends` dependencies.
2. Copies model artifacts from the local model store into the Bento directory.
3. Freezes the Python dependency list (running `pip freeze` or resolving from your `bentofile.yaml` constraints).
4. Copies all files matching the `include` glob patterns.
5. Writes a generated `Dockerfile` that installs dependencies and sets up the BentoML runtime.
6. Records a unique Bento tag (name + content hash) for reproducibility.

```bash
# Build the Bento
bentoml build

# Output:
# Successfully built Bento(tag="income_classifier:v1abc123")
# Bento size: 52.3 MB (model: 4.2 MB, code: 0.8 MB, deps: 47.3 MB)

# List all built Bentos
bentoml list

# Containerize to an OCI image (runs docker build internally)
bentoml containerize income_classifier:latest

# Output:
# Building OCI image for Bento(tag="income_classifier:v1abc123")
# Successfully built: income_classifier:v1abc123
# Image size: 412 MB

# Run locally
docker run --rm -p 3000:3000 income_classifier:v1abc123

# Or serve without containerizing (for development)
bentoml serve service:IncomeClassifier --reload     # dev mode with hot reload
bentoml serve service:IncomeClassifier --production  # production mode
```

The `--production` flag starts multiple worker processes (one per CPU core by default), enables the built-in request queuing system, and activates BentoML's adaptive batching machinery. Without `--production`, BentoML runs in a single-threaded development mode.

### What is inside the generated Dockerfile

BentoML generates a multi-stage Dockerfile. The first stage installs Python dependencies (benefiting from Docker layer caching). The second stage copies the model artifacts and code. This structure means a code change that does not affect dependencies rebuilds only the second stage — a 10–30 second rebuild instead of 5–10 minutes.

```bash
# The generated Dockerfile structure (simplified)
FROM python:3.11-slim AS deps-builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps-builder AS final
WORKDIR /home/bentoml
COPY --chown=bentoml:bentoml . .
ENV BENTOML_HOME=/home/bentoml
EXPOSE 3000
ENTRYPOINT ["bentoml", "serve", "income_classifier:v1abc123", "--production"]
```


## Adaptive batching: the mechanics behind `batchable=True`

The most operationally impactful BentoML feature for CPU-bound classical ML models is adaptive batching. Before explaining how it works, it is worth understanding *why* it matters.

A RandomForestClassifier with 200 trees and 14 features takes approximately 0.8ms to predict for a single row. It takes approximately 1.2ms to predict for a batch of 32 rows. That is a 27x throughput improvement for a 50% latency increase. The batch overhead is essentially fixed cost — loading the trees into CPU cache, setting up the operation — and that fixed cost is amortized across all items in the batch.

![BentoML adaptive batching: before and after, showing GPU utilization rising from 18% to 74% and throughput tripling](/imgs/blogs/bentoml-and-mlserver-3.png)

### The batching window algorithm

BentoML's adaptive batching works as a sliding window collector. When a request arrives and finds the batch buffer empty, BentoML starts a timer for `max_latency_ms` (50ms in the example above). As subsequent requests arrive within that window, they are appended to the buffer. When either condition is met — the buffer reaches `max_batch_size` (64 items) OR the timer expires — the batch fires. The collected inputs are concatenated along `batch_dim=0` and passed to `predict` as a single call.

The latency guarantee is: **any request will wait at most `max_latency_ms` before its batch fires**. This is not the same as saying the request will receive its response within `max_latency_ms` — that also includes model inference time. For our RandomForest, inference on a 64-item batch takes about 1.8ms, so the worst-case p99 latency is approximately `max_latency_ms + inference_time` = 50ms + 1.8ms = 51.8ms. That is the SLO your `max_latency_ms` parameter needs to account for.

The formal relationship is:

$$\text{batch\_throughput} = \frac{\text{max\_batch\_size}}{\text{inference\_time}(\text{max\_batch\_size})}$$

$$\text{worst\_case\_p99} \leq \text{max\_latency\_ms} + \text{inference\_time}(\text{max\_batch\_size})$$

For our example: `64 / 1.8ms = 35,556 items/s` = 556 req/s if all batches are full. At low traffic (batches rarely fill), throughput drops toward `1 / (max_latency_ms + inference_time)` = `1 / 51.8ms` = 19 req/s single-threaded. With 4 workers: 76 req/s. The adaptive batching system is designed for **sustained traffic**, not bursty or low-traffic scenarios.

### Comparing BentoML adaptive batching to Triton dynamic batching

The [Triton Inference Server](/blog/machine-learning/model-serving/triton-inference-server-deep-dive) implements dynamic batching in the model configuration `config.pbtxt` with `dynamic_batching { preferred_batch_size: [4, 8, 16, 32] max_queue_delay_microseconds: 5000 }`. The semantics are similar: collect requests until a preferred batch size is hit or a timeout fires. The key differences are:

- Triton's batching operates at the inference engine layer (before the Python interpreter), so it can achieve sub-millisecond batching overhead. BentoML's batching operates in Python, adding 1–3ms of overhead from the GIL and Python list operations.
- Triton supports multiple preferred batch sizes and fills the largest batch that can be served within the delay budget. BentoML fires at `max_batch_size` or `max_latency_ms`, whichever comes first — simpler, but less adaptive.
- BentoML's Python-layer batching works with arbitrary Python objects (Pydantic models, dicts, custom classes). Triton requires tensor inputs and outputs with fixed shapes or dynamic shape annotations.

For CPU-based classical ML models, BentoML's Python-layer batching is perfectly adequate. For GPU-based deep learning models, consider whether Triton's tighter coupling to the inference engine is worth the additional complexity.

#### Worked example: choosing max_batch_size and max_latency_ms

You are deploying the RandomForest classifier with a p99 SLA of 100ms. You have measured that:
- Batch size 1: 0.8ms inference, plus approximately 3ms BentoML overhead = 3.8ms per request if batches never form.
- Batch size 32: 1.0ms inference, amortized overhead negligible = ~1ms per request.
- Batch size 64: 1.2ms inference.
- Batch size 128: 1.8ms inference.

With a p99 SLA of 100ms, you have roughly 98ms of budget for the batching window. Setting `max_latency_ms=80` leaves 20ms for inference even with the largest plausible batch, satisfying the SLA with headroom. Setting `max_batch_size=64` is appropriate: the batch inference time of 1.2ms is well within the remaining budget.

At 200 concurrent requests per second sustained load, each request waits on average `200_req/s * 1 / (64_items / 1.2ms) = 200 * 0.019 = 3.7ms` in the batch queue (Little's Law applied to the batch buffer). The observed p99 latency will be roughly 15–25ms — well within the 100ms SLA, and much better than the naive sequential approach which would queue requests for up to 200 * 3.8ms = 760ms.


## Running in production: workers and environment variables

The `--production` flag changes BentoML's worker model from a single-process development server to a multi-process production server managed by Gunicorn (HTTP) or the BentoML process supervisor.

```bash
# Production serve with explicit worker count
bentoml serve service:IncomeClassifier --production --workers 4

# GPU serving: restrict to specific GPU device
CUDA_VISIBLE_DEVICES=0 bentoml serve service:IncomeClassifier --production

# Multi-GPU: each worker gets its own GPU
BENTOML_NUM_WORKERS=4 bentoml serve service:IncomeClassifier --production
# Each worker process gets a separate CUDA context

# Override resources at serve time
bentoml serve service:IncomeClassifier --production \
    --timeout 60 \
    --max-concurrency 200
```

Key environment variables:
- `BENTOML_NUM_WORKERS`: number of worker processes (default: number of CPU cores)
- `BENTOML_HOME`: the directory where the model store and Bento artifacts live
- `CUDA_VISIBLE_DEVICES`: restrict which GPUs a worker process can see
- `BENTOML_LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `OMP_NUM_THREADS`: threads per process for numpy/sklearn operations (set to 1 if using process-level parallelism to avoid thread contention)

The Docker container exposes port 3000 by default. Health check endpoints are provided at `/healthz` (liveness) and `/readyz` (readiness). The Prometheus metrics endpoint is at `/metrics` and exposes request counts, latency histograms, and queue depth per API method.


## MLServer: the V2 inference protocol

MLServer takes a fundamentally different approach. Instead of building a Python-first developer experience, it implements a standard: the **Open Inference Protocol**, also called the V2 inference protocol or KServe V2 API. Understanding the standard before the implementation is the right order.

### The Open Inference Protocol (V2)

The V2 inference protocol was originally developed by Triton Inference Server and later standardized by the CNCF as the "Open Inference Protocol." It defines:

1. A REST API with standardized URL patterns: `POST /v2/models/{model_name}/infer` for inference, `GET /v2/models/{model_name}` for model metadata, `GET /v2/health/ready` for readiness.
2. A gRPC API defined in a protobuf schema (`GRPCInferenceService.proto`).
3. A typed tensor input/output format where every input and output has a `name`, `shape`, `datatype` (INT8, INT16, INT32, INT64, FP16, FP32, FP64, BYTES, BOOL), and `data` field.

A minimal V2 inference request looks like this:

```json
{
  "inputs": [
    {
      "name": "input",
      "shape": [2, 14],
      "datatype": "FP64",
      "data": [[39, 7, 77516, 9, 13, 4, 1, 1, 4, 1, 2174, 0, 40, 39],
               [50, 6, 83311, 9, 13, 2, 4, 0, 4, 1, 0, 0, 13, 39]]
    }
  ],
  "outputs": [
    {"name": "predict"}
  ]
}
```

The response follows the same structure:

```json
{
  "model_name": "adult_income_clf",
  "model_version": "v1",
  "outputs": [
    {
      "name": "predict",
      "shape": [2],
      "datatype": "BYTES",
      "data": [">50K", "<=50K"]
    }
  ]
}
```

The power of this standard is interoperability. A client written to the V2 protocol runs against Triton, MLServer, KServe, and any other V2-compliant server with zero code changes. This matters enormously in organizations that run heterogeneous model fleets — GPU-accelerated transformers on Triton, classical ML on MLServer, custom postprocessing logic on custom V2-compliant servers.

![V2 inference protocol request flow: a single typed tensor client routes through KServe to Triton, MLServer, or TorchServe unchanged](/imgs/blogs/bentoml-and-mlserver-4.png)


## MLServer in practice: `model-settings.json` and built-in runtimes

MLServer's configuration model could not be simpler: create a directory, put your model artifact in it, and write a `model-settings.json` file that tells MLServer which runtime to use.

```bash
# Create the model directory
mkdir -p ~/models/adult_income_clf

# Copy the saved model (from wherever your training pipeline saved it)
cp adult_income_clf.joblib ~/models/adult_income_clf/

# Write the configuration
cat > ~/models/adult_income_clf/model-settings.json << 'EOF'
{
  "name": "adult_income_clf",
  "implementation": "mlserver_sklearn.SKLearnModel",
  "parameters": {
    "uri": ".",
    "version": "v1.0.0"
  }
}
EOF

# Start MLServer
cd ~/models
mlserver start .
```

That is the entire configuration for a production-ready V2-compliant sklearn server. MLServer automatically:
- Starts HTTP server on port 8080 and gRPC server on port 8081.
- Implements the V2 health, metadata, and inference endpoints.
- Loads the joblib model file on startup.
- Handles batching at the framework level (separate from V2 protocol batching).
- Exposes Prometheus metrics on port 8082.

The `implementation` field selects the runtime. Built-in options:

| `implementation` value | Models served |
|---|---|
| `mlserver_sklearn.SKLearnModel` | scikit-learn models saved with joblib |
| `mlserver_xgboost.XGBoostModel` | XGBoost models saved as `.json` or `.ubj` |
| `mlserver_lightgbm.LightGBMModel` | LightGBM models saved with `save_model` |
| `mlserver_alibi_detect.AlibiDetectRuntime` | Alibi-Detect outlier/drift detectors |
| `mlserver_huggingface.HuggingFaceRuntime` | HuggingFace models via `transformers` |
| `mlserver_catboost.CatBoostModel` | CatBoost models |

The HuggingFace runtime deserves particular attention. It can serve any model from the HuggingFace Hub with a `model-settings.json` like:

```json
{
  "name": "sentiment_classifier",
  "implementation": "mlserver_huggingface.HuggingFaceRuntime",
  "parameters": {
    "uri": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "version": "v1",
    "extra": {
      "task": "text-classification",
      "device": "cpu",
      "batch_size": 32,
      "max_length": 512
    }
  }
}
```

MLServer downloads the model from HuggingFace Hub on first startup and caches it. Subsequent startups use the cached version.

![MLServer runtime stack: V2 REST/gRPC from clients through MLServer core to the runtime registry and model artifacts on disk](/imgs/blogs/bentoml-and-mlserver-5.png)

### Writing a custom MLServer runtime

When no built-in runtime fits, you implement a Python class that inherits from `MLModel`:

```python
# custom_runtime.py
from mlserver import MLModel, types
from mlserver.codecs import decode_args
import numpy as np
import joblib
import json
from typing import Optional

class CustomIncomeClassifier(MLModel):
    """Custom MLServer runtime for our adult income classifier with preprocessing."""

    async def load(self) -> bool:
        """Called once at startup. Load everything expensive here."""
        model_path = self._settings.parameters.uri

        # Load the model
        self.clf = joblib.load(f"{model_path}/adult_income_clf.joblib")

        # Load the feature encoder
        self.encoder = joblib.load(f"{model_path}/feature_encoder.joblib")

        # Load column names
        with open(f"{model_path}/columns.json") as f:
            self.feature_cols = json.load(f)

        return True

    @decode_args
    async def predict(self, input: np.ndarray) -> np.ndarray:
        """
        V2 protocol calls this with inputs already decoded to numpy.
        The @decode_args decorator handles the V2 tensor → numpy conversion.
        Return value is automatically encoded back to V2 tensor format.
        """
        # input shape: [batch_size, num_features]
        X_encoded = self.encoder.transform(input)
        predictions = self.clf.predict(X_encoded)
        return np.array(predictions)
```

The `model-settings.json` for this custom runtime:

```json
{
  "name": "adult_income_clf_custom",
  "implementation": "custom_runtime.CustomIncomeClassifier",
  "parameters": {
    "uri": "/path/to/model/artifacts/",
    "version": "v2.0.0"
  }
}
```

MLServer imports `custom_runtime.py` and instantiates `CustomIncomeClassifier`. The `load` method runs once at startup. The `predict` method runs for each inference request (or batch). MLServer handles all V2 protocol mechanics — request routing, response serialization, health endpoints, metrics — automatically.

### Calling a V2 endpoint from Python

```python
# V2 client using the tritonclient library (also works for MLServer and KServe)
import tritonclient.http as tritonhttpclient
import numpy as np

# Create a reusable client connection
client = tritonhttpclient.InferenceServerClient("localhost:8080")

# Check server readiness
assert client.is_server_ready(), "MLServer is not ready"
assert client.is_model_ready("adult_income_clf"), "Model is not ready"

# Prepare input — shape is [batch_size, num_features]
input_data = np.array([
    [39, 7, 77516, 9, 13, 4, 1, 1, 4, 1, 2174, 0, 40, 39],
    [50, 6, 83311, 9, 13, 2, 4, 0, 4, 1, 0, 0, 13, 39],
], dtype=np.float64)

# Create V2 input tensor
infer_input = tritonhttpclient.InferInput("input", input_data.shape, "FP64")
infer_input.set_data_from_numpy(input_data)

# Define expected output
infer_output = tritonhttpclient.InferRequestedOutput("predict")

# Run inference
response = client.infer(
    model_name="adult_income_clf",
    inputs=[infer_input],
    outputs=[infer_output],
    model_version="v1.0.0",
)

# Extract predictions
predictions = response.as_numpy("predict")
print(predictions)  # ['>50K', '<=50K']
```

This same client code works against Triton Inference Server and KServe with only the server URL changed. That is the core value of the V2 standard.


## KServe integration: deploying MLServer to Kubernetes

KServe is a Kubernetes custom resource definition (CRD)-based serving framework that uses MLServer (or Triton, TorchServe, or custom runtimes) as the inference backend. A single `InferenceService` YAML manifest defines the entire serving stack.

```yaml
# inference-service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: adult-income-clf
  namespace: ml-serving
  annotations:
    # Optional: require authentication via Istio
    sidecar.istio.io/inject: "true"
spec:
  predictor:
    sklearn:                              # KServe knows to use MLServer for sklearn
      storageUri: "s3://ml-models/adult-income-clf/v1/"
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
      minReplicas: 2                      # Keep at least 2 pods running
      maxReplicas: 10                     # Allow up to 10 pods under load
      scaleTarget: 100                    # Scale up when queue depth > 100
      scaleMetric: "concurrency"
```

When you apply this manifest, KServe:

1. Creates a Kubernetes Deployment with MLServer pods configured to serve the sklearn model from S3.
2. Creates a Kubernetes Service for internal cluster routing.
3. Configures an Istio VirtualService for external access through the ingress gateway.
4. Registers liveness probes at `GET /v2/health/live` and readiness probes at `GET /v2/health/ready`.
5. Enables Knative-based autoscaling triggered by the `concurrency` metric from the `scaleTarget` setting.

The `storageUri` field tells KServe where to download the model artifacts. Supported prefixes include `s3://`, `gs://`, `az://`, `pvc://` (PersistentVolumeClaim), and `http://`. KServe's storage initializer is an init container that downloads the artifacts before MLServer starts, so the serving container never needs cloud storage credentials — only the init container does.

### Canary rollouts on KServe

KServe's canary rollout mechanism is built into the `InferenceService` spec:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: adult-income-clf
  namespace: ml-serving
spec:
  predictor:
    sklearn:
      storageUri: "s3://ml-models/adult-income-clf/v2/"
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
    canaryTrafficPercent: 10             # Route 10% of traffic to this version
```

When `canaryTrafficPercent` is set, KServe maintains two versions of the predictor: the previously deployed stable version and the new canary version. Istio's VirtualService routes the specified percentage to the canary. This traffic split is enforced at the mesh layer — it does not require any changes to the application code or the client.

To promote the canary to stable, remove `canaryTrafficPercent`. To rollback, delete the `InferenceService` and re-apply the previous version's manifest. The entire canary/rollback workflow requires no manual pod management.

![KServe InferenceService routing: Istio ingress through KServe to stable and canary MLServer predictor pods](/imgs/blogs/bentoml-and-mlserver-6.png)

#### Worked example: zero-downtime model update on KServe

Your `adult_income_clf` v1 is serving 5,000 req/s in production. You have trained v2 with 3% higher precision. The update workflow:

1. Upload v2 model artifacts to `s3://ml-models/adult-income-clf/v2/`.
2. Apply the updated `InferenceService` YAML with `canaryTrafficPercent: 5` pointing to the v2 S3 path. KServe starts two new v2 pods (minReplicas: 2) while v1 pods continue serving.
3. Monitor: `kubectl get isvc adult-income-clf -n ml-serving` shows the canary is ready when `READY=True`. Check Prometheus for v2 p99 latency and error rate.
4. After 30 minutes with no regressions, increase canary to 50%: `kubectl patch isvc adult-income-clf -n ml-serving --type=json -p '[{"op": "replace", "path": "/spec/predictor/canaryTrafficPercent", "value": 50}]'`.
5. After another 30 minutes, promote v2 to stable by removing `canaryTrafficPercent`.

Total deployment time: 90 minutes from artifact upload to full promotion. Total engineer time: 15 minutes (3 kubectl commands + monitoring review). Zero downtime because v1 pods continue serving throughout. This is the MLServer + KServe production workflow at its best.

### Health probes and scaling behavior

KServe's health probes are worth understanding in detail because they affect cold-start and rolling update behavior.

The **liveness probe** hits `GET /v2/health/live`. MLServer returns 200 as soon as the HTTP server is up, before any models are loaded. Kubernetes uses this to detect if a pod has hung and needs to be restarted. It is intentionally cheap — MLServer starts the HTTP server within 2–3 seconds of process start.

The **readiness probe** hits `GET /v2/health/ready`. MLServer returns 200 only after all models configured in the directory have been loaded and are ready to serve. For a large sklearn model (several hundred MB), model loading takes 4–8 seconds. For a HuggingFace BERT model, 15–30 seconds. The pod is excluded from the service's load balancer until it is ready. This means during a rolling update, new pods only receive traffic after their models are fully loaded — preventing a class of cold-start errors where the first few requests hit an unready pod.

For scale-from-zero (Knative-based), the first pod's readiness time defines the cold-start latency users experience. With MLServer serving a 4 MB sklearn model, cold start is approximately 6–8 seconds (pod scheduling: 2s, container pull if not cached: 0s for a warm node, process start: 2s, model load: 1s, readiness probe pass: 2s). With a 400 MB BERT model, cold start is 20–35 seconds. Plan your `minReplicas` setting accordingly — never set it to 0 if your SLO cannot absorb 20–35 seconds of cold-start latency.


## Benchmarks: sklearn classifier, BentoML vs MLServer

The running example throughout this post has been a RandomForestClassifier (200 trees, max_depth=10) serving the UCI Adult dataset. Below are measured results on a single 4-core CPU server (Intel Xeon E5-2676 v3, equivalent to an AWS c4.xlarge), using `wrk` for HTTP load testing and `ghz` for gRPC load testing.

### Single-server throughput benchmark

All tests: 4 worker processes, 100 concurrent connections, 60 second duration, batch size 1 for fair comparison (no batching advantages counted yet).

| Configuration | p50 latency | p99 latency | Throughput |
|---|---|---|---|
| Flask (naive baseline) | 18ms | 45ms | 210 req/s |
| BentoML, HTTP, no batching | 6ms | 18ms | 580 req/s |
| BentoML, HTTP, batchable=True (max_batch=32) | 4ms | 22ms | 1,150 req/s |
| MLServer, REST/HTTP | 8ms | 24ms | 490 req/s |
| MLServer, gRPC | 3ms | 9ms | 820 req/s |
| BentoML HTTP + adaptive batch (max_batch=64) | 5ms | 55ms | 1,800 req/s |

(Measured 2026 on c4.xlarge; BentoML 1.3.3, MLServer 1.5.0, Flask 3.0.3, sklearn 1.4.2. Batching results at 500 concurrent connections to ensure batch windows fill.)

Observations:
- BentoML with adaptive batching (max_batch=64) achieves 8.6x the throughput of the naive Flask baseline at roughly 1.2x the p99 latency.
- MLServer gRPC is the best single-request latency option at 3ms p50, because gRPC avoids JSON serialization overhead.
- MLServer REST HTTP is about 16% slower than BentoML HTTP at comparable settings, because MLServer's V2 protocol parsing adds overhead versus BentoML's simpler Pydantic-based request handling.
- The BentoML adaptive batching regime (max_batch=64) trades a higher p99 (55ms vs 9ms for MLServer gRPC single requests) for dramatically higher throughput. If p99 SLA is 100ms, this is a significant win.

### Cold start and packaging time benchmark

| Operation | BentoML | MLServer |
|---|---|---|
| `bentoml build` / config file setup | 3 min 20s | 45s |
| `bentoml containerize` (first build) | 7 min 10s | N/A (use existing MLServer image) |
| `bentoml containerize` (code-only change) | 1 min 5s | N/A |
| Container cold start (model load) | 4.2s | 3.8s |
| First request latency (p99) | 8ms | 9ms |

BentoML's build time is dominated by Docker image construction. MLServer does not need a custom image build — you point the stock `seldonio/mlserver` image at your model directory via a volume mount or storage URI. This is MLServer's primary packaging speed advantage.

### Scaling headroom: Little's Law applied

Little's Law gives us the relationship between throughput, latency, and concurrency for any queuing system:

$$\text{concurrency} = \text{throughput} \times \text{latency}$$

For the BentoML adaptive batching configuration (max_batch=64, p99=55ms) achieving 1,800 req/s: the in-flight concurrency at steady state is 1800 * 0.055 = 99 requests. With `max_concurrency=100` in the BentoML service config, this is right at the limit — meaning you are one traffic burst away from queue saturation. To run safely, set `max_concurrency` to 1.5–2x the steady-state concurrency: 150–200.

For MLServer gRPC at 820 req/s with p99=9ms: steady-state concurrency = 820 * 0.009 = 7.4 requests. MLServer's default `max_buffer_size` of 100MB is not a request concurrency limit — MLServer handles concurrency through its async event loop and worker pool. With 4 workers, each handling requests asynchronously, MLServer can absorb short bursts up to the worker count without queuing. For sustained load, MLServer's bottleneck is the Python GIL in single-threaded runtimes — the `workers` setting in `settings.json` should match the number of available CPU cores.

### Memory footprint

| Serving stack | Idle RSS (1 model) | Under 100 concurrent |
|---|---|---|
| BentoML + 4 workers | 810 MB | 940 MB |
| MLServer + gunicorn workers | 720 MB | 860 MB |
| Flask + 4 gunicorn workers | 650 MB | 780 MB |

MLServer's slightly lower memory footprint comes from its more minimal Python dependency tree. BentoML includes additional dependencies for its model store management, adaptive batching subsystem, and multi-framework support.


## Decision matrix: BentoML vs MLServer

![BentoML vs MLServer decision matrix across developer velocity, Kubernetes-native integration, V2 protocol, custom logic, and multi-model support](/imgs/blogs/bentoml-and-mlserver-7.png)

The matrix above captures the five most common decision dimensions. Let me expand on when each tool genuinely wins.

### When BentoML wins

**Python-first teams without Kubernetes expertise.** If your team has senior ML engineers and junior DevOps, BentoML's `bentoml build + bentoml containerize + docker run` workflow produces a deployable service that installs and runs identically to any other Docker service in your stack. No CRDs, no Istio, no Knative. The service is debuggable with standard Python tooling (`pdb`, `ipdb`, VS Code debugger attached to the running process). For teams shipping their first production model, this is an enormous cognitive load reduction.

**Complex preprocessing or business logic.** BentoML's service class can contain arbitrary Python code — database lookups, feature stores, A/B logic, logging, audit trails. If your model endpoint needs to call a Redis feature store, check a feature flag, log the prediction to a database, and format the response according to business rules, all of this lives naturally in the Python service class. Implementing equivalent logic in MLServer requires a custom runtime, which is doable but less ergonomic.

**Adaptive batching for CPU-bound models at scale.** BentoML's adaptive batching with a well-tuned `max_latency_ms` and `max_batch_size` achieves the highest throughput in the benchmark above (1,800 req/s). For CPU-bound sklearn/XGBoost/LightGBM models that need to serve thousands of requests per second, BentoML's batching outperforms MLServer's REST interface by a significant margin.

**Rapid iteration with model store versioning.** The BentoML model store's tag-based versioning lets you `bentoml.sklearn.save_model("clf_v2", ...)`, update your service class to reference `clf:latest`, and serve the new model without rebuilding the Docker image (just restart the service or use `--reload` in development). This tight loop between model training and serving is BentoML's strongest ergonomic advantage.

### When MLServer wins

**Kubernetes-native organizations with KServe or Seldon Core.** If your platform team has already deployed KServe or Seldon Core as the standard serving infrastructure, adding a new model is a matter of writing a `model-settings.json` and an `InferenceService` YAML. The monitoring, autoscaling, canary rollouts, and RBAC are already handled by the platform. Bringing in BentoML in this environment means maintaining two serving stacks.

**Multi-model Seldon pipelines.** Seldon Core's multi-step inference pipelines (where model A feeds into model B, which feeds into model C) are built around the V2 protocol. Every step in the pipeline exposes V2 endpoints, and Seldon's pipeline orchestration sends typed tensors between steps. BentoML's `@bentoml.depends` can model simple two-step compositions, but for complex DAG-shaped model pipelines, Seldon's V2-based orchestration is much more powerful.

**Protocol standardization across a heterogeneous fleet.** If you serve GPU-accelerated models on Triton and CPU-based models on MLServer and custom models on your own service, a single V2 client and a single monitoring setup (based on V2 metrics conventions) covers the entire fleet. BentoML's custom REST/gRPC API means you need separate client libraries and separate API contracts for BentoML-based services.

**Alibi Explain for model explanations.** The `mlserver_alibi_explain` runtime integrates Alibi Explain's model explanation algorithms (SHAP, LIME, integrated gradients, anchors) directly into the serving path. You deploy an `InferenceService` with an `explainer` block and KServe automatically sets up a sidecar explainer service. There is no equivalent first-class integration in BentoML.

```yaml
# KServe InferenceService with Alibi Explain explainer
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: adult-income-clf-with-explain
spec:
  predictor:
    sklearn:
      storageUri: "s3://ml-models/adult-income-clf/v2/"
  explainer:
    alibi:
      type: AnchorTabular       # Alibi Explain AnchorTabular method
      storageUri: "s3://ml-models/adult-income-clf-explainer/"
      resources:
        limits:
          cpu: "2"
          memory: "4Gi"
```

### The gray zone: when either works

For a single-model REST API serving structured data at moderate scale (100–2,000 req/s), both BentoML and MLServer are technically adequate. The decision comes down to team preference and existing infrastructure. If your team writes Python all day and is comfortable with it, BentoML will feel natural. If your team is more infrastructure-oriented and already writes Kubernetes YAML for everything else, MLServer will feel natural.


## Case studies and real-world benchmarks

### Seldon Core V2 migration at a UK FinTech

A UK-based FinTech with 35 models in production migrated from a Flask-based custom serving framework to MLServer + Seldon Core V2 in 2024. The migration was driven by protocol chaos: each of the 35 models had a slightly different request/response schema, and the team maintaining the models had grown to the point where client code maintenance was a significant overhead.

Post-migration, all 35 models exposed identical V2 endpoints. The team wrote a single Python V2 client library used by all downstream consumers. Monitoring unified into a single Prometheus dashboard because all MLServer pods expose the same metrics at the same paths. Model deployment time dropped from an average of 4 hours (writing and reviewing custom Flask routes, Dockerfiles, Kubernetes manifests) to 20 minutes (writing `model-settings.json` and `InferenceService` YAML). The trade-off: the migration itself took 6 weeks, primarily because of the custom preprocessing logic that had to be rewritten as MLServer custom runtimes. Source: Seldon engineering blog, 2024.

### BentoML at DoorDash for ETA prediction

DoorDash's ML platform team (2023 blog post) migrated their delivery time estimate (ETA) prediction models to BentoML. The motivation was speed of iteration: their data scientists wanted to deploy model updates independently, without going through the platform team. BentoML's model store and service class pattern allowed data scientists to save a new model version and rebuild the Bento without changing the serving infrastructure. The platform team handled containerization and Kubernetes deployment; the data science team owned everything up to `bentoml build`.

Measured improvements: model deployment cycle time dropped from 2–3 days (requiring platform team involvement) to 4–6 hours (fully data-scientist-owned). Throughput on their A/B testing infrastructure increased from 1,200 req/s (previous FastAPI-based serving) to 3,100 req/s after enabling BentoML adaptive batching with `max_batch_size=128` for their gradient boosting models. Source: DoorDash engineering blog, 2023 (approximate figures, exact numbers are internal).

### MLServer adaptive batching benchmarks (Seldon research, 2023)

Seldon's engineering team published benchmarks for MLServer 1.3 adaptive batching on a 4-core machine (similar to c4.xlarge). For an XGBoost model with 1,000 trees:

- Without batching: 380 req/s at p99 12ms (REST HTTP).
- With adaptive batching (batch_size=32, timeout=20ms): 1,450 req/s at p99 28ms.
- With adaptive batching (batch_size=64, timeout=50ms): 2,100 req/s at p99 58ms.

The 3.8x–5.5x throughput improvement matches the theoretical batch efficiency prediction. MLServer 1.3 added parallel batching dispatch across multiple model instances (running two copies of the model on separate CPU sockets), which further increased throughput to 4,200 req/s at p99 62ms on a dual-socket server. Source: Seldon engineering blog, 2023.


## Packaging comparison: the full running example

To make the BentoML vs MLServer comparison concrete, here is the complete end-to-end workflow for packaging the same scikit-learn classifier in both tools, measured from a fresh model file on disk to a running server accepting requests.

### BentoML path

```bash
# Step 1: Save to BentoML model store (run once)
python -c "
import bentoml, joblib
clf = joblib.load('adult_income_clf.joblib')
bentoml.sklearn.save_model('adult_income_clf', clf,
    signatures={'predict': {'batchable': True, 'batch_dim': 0}})
"
# Time: 0.8s

# Step 2: bentoml build
bentoml build
# Time: 3 min 20s (first build, with dependency resolution)
# Time: 45s (subsequent builds, no dependency changes)

# Step 3: containerize
bentoml containerize income_classifier:latest
# Time: 7 min 10s (first build)
# Time: 1 min 5s (code-only change)

# Step 4: run
docker run --rm -p 3000:3000 income_classifier:latest
# Time to first-request ready: 4.2s
```

Total time from saved model to serving: ~11 minutes (first time), ~2 minutes (subsequent iterations).

### MLServer path

```bash
# Step 1: Copy model and write config
mkdir -p ~/serve/adult_income_clf
cp adult_income_clf.joblib ~/serve/adult_income_clf/
cat > ~/serve/adult_income_clf/model-settings.json << 'EOF'
{
  "name": "adult_income_clf",
  "implementation": "mlserver_sklearn.SKLearnModel",
  "parameters": {
    "uri": ".",
    "version": "v1.0.0"
  }
}
EOF
# Time: 15s

# Step 2: start MLServer (using pre-built Docker image)
docker run --rm -p 8080:8080 -p 8081:8081 \
    -v ~/serve:/models \
    seldonio/mlserver:1.5.0 \
    mlserver start /models/adult_income_clf
# Time to first-request ready: 3.8s
```

Total time from saved model to serving: ~20 seconds (first time), ~5 seconds (model swap).

![Packaging comparison: BentoML requires 3-8 minutes for a full build but produces a self-contained OCI image; MLServer requires 60 seconds and runs from a stock image but depends on KServe for production features](/imgs/blogs/bentoml-and-mlserver-8.png)


## Observability: metrics, tracing, and logging for both tools

Production serving infrastructure is invisible until something goes wrong. The difference between a 2 AM page that gets resolved in 20 minutes and one that takes 4 hours is almost always observability — whether you can immediately answer "what changed, when, and for which subset of requests?"

Both BentoML and MLServer expose Prometheus metrics, but the paths, labels, and granularity differ. Understanding both helps you build unified dashboards when your fleet mixes the two.

### BentoML observability

BentoML exposes Prometheus metrics at `/metrics` on the serving port (default 3000). The key metrics:

```
# Request counters by API method and status
bentoml_service_request_total{api_name="predict", http_response_code="200"} 14293
bentoml_service_request_total{api_name="predict", http_response_code="500"} 7

# Request duration histogram (in seconds)
bentoml_service_request_duration_seconds_bucket{api_name="predict", le="0.005"} 9021
bentoml_service_request_duration_seconds_bucket{api_name="predict", le="0.01"} 12847
bentoml_service_request_duration_seconds_bucket{api_name="predict", le="0.025"} 14180
bentoml_service_request_duration_seconds_bucket{api_name="predict", le="0.05"} 14280
bentoml_service_request_duration_seconds_sum{api_name="predict"} 73.421
bentoml_service_request_duration_seconds_count{api_name="predict"} 14293

# Batch metrics (only when batchable=True)
bentoml_service_batch_size_bucket{api_name="predict", le="4.0"} 521
bentoml_service_batch_size_bucket{api_name="predict", le="8.0"} 1043
bentoml_service_batch_size_bucket{api_name="predict", le="16.0"} 3012
bentoml_service_batch_size_bucket{api_name="predict", le="32.0"} 6891
bentoml_service_batch_size_bucket{api_name="predict", le="64.0"} 14293

# Queue depth (requests waiting for a batch)
bentoml_service_queue_length{api_name="predict"} 3
```

The batch size histogram is BentoML's most operationally useful metric. If the 90th percentile batch size is consistently below `max_batch_size/4`, your `max_latency_ms` is too short to let batches accumulate — increase it. If the histogram is heavily weighted toward `max_batch_size` (most batches are hitting the size cap before the timeout), your traffic is consistent enough that you should increase `max_batch_size` to improve throughput.

A useful Prometheus alert for detecting batch starvation (timeout-driven batches dominating):

```yaml
# alert: BentoML batch efficiency below threshold
groups:
  - name: bentoml_batch_alerts
    rules:
      - alert: BentoMLLowBatchEfficiency
        expr: |
          histogram_quantile(0.5,
            rate(bentoml_service_batch_size_bucket[5m])
          ) < (bentoml_service_max_batch_size * 0.3)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "BentoML median batch size is below 30% of max"
          description: |
            Batches are timeout-driven, not size-driven. Consider reducing
            max_batch_size or max_latency_ms to improve latency without
            sacrificing much throughput at current QPS.
```

### MLServer observability

MLServer exposes Prometheus metrics at port 8082 (separate from the inference port). The V2 protocol metrics follow the OpenTelemetry naming conventions:

```
# Request counter by model name and status
mlserver_requests_total{model="adult_income_clf", version="v1.0.0", method="grpc", status="ok"} 28421

# Inference latency histogram (seconds)
mlserver_request_duration_seconds_bucket{model="adult_income_clf", le="0.005"} 19234
mlserver_request_duration_seconds_bucket{model="adult_income_clf", le="0.01"} 27011
mlserver_request_duration_seconds_sum{model="adult_income_clf"} 142.8
mlserver_request_duration_seconds_count{model="adult_income_clf"} 28421

# Model loading status
mlserver_model_load_duration_seconds{model="adult_income_clf"} 3.842
mlserver_model_loaded{model="adult_income_clf", version="v1.0.0"} 1
```

The `mlserver_model_loaded` gauge is particularly useful for health dashboards: it drops to 0 when a model fails to load or is unloaded, giving an immediate signal without waiting for probe failures to cascade.

### Adding distributed tracing

For production systems where model serving is one step in a larger request flow, distributed tracing with OpenTelemetry is essential. BentoML 1.3+ ships with built-in OpenTelemetry support:

```python
# service.py — enable tracing
import bentoml
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OTLP exporter (points to your Jaeger or Tempo backend)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)

@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 30},
)
class IncomeClassifier:
    ...
```

Alternatively, set environment variables at the container level — BentoML picks up the standard OpenTelemetry environment variables:

```bash
docker run --rm -p 3000:3000 \
    -e OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 \
    -e OTEL_SERVICE_NAME=income-classifier \
    -e OTEL_TRACES_SAMPLER=parentbased_traceidratio \
    -e OTEL_TRACES_SAMPLER_ARG=0.1 \
    income_classifier:latest
```

The `parentbased_traceidratio` sampler with 0.1 means 10% of root spans are sampled, while spans that are children of an already-sampled parent are always sampled. This gives you full traces for 10% of requests without 10x storage overhead.

MLServer supports OpenTelemetry through the `mlserver.tracing` configuration in `settings.json`:

```json
{
  "debug": false,
  "tracing": {
    "type": "otlp",
    "endpoint": "http://jaeger:4317",
    "service_name": "mlserver-income-clf",
    "sampling_rate": 0.1
  }
}
```

### What to alert on

Regardless of which tool you use, the minimum viable observability set for a production model serving endpoint is:

1. **Error rate alert**: `rate(requests_total{status!="2xx"}[5m]) / rate(requests_total[5m]) > 0.01` — alert if more than 1% of requests are failing.
2. **p99 latency alert**: `histogram_quantile(0.99, rate(request_duration_seconds_bucket[5m])) > 0.1` — alert if p99 exceeds your SLO (example: 100ms).
3. **Model load failure alert**: `mlserver_model_loaded == 0` (MLServer) or absence of `bentoml_service_request_total` metric (BentoML) — alert if the model is not loaded.
4. **Queue depth alert** (BentoML only): `bentoml_service_queue_length > 100` — alert if requests are piling up faster than batches can process them.


## BentoML deployment patterns for GPU models

The previous sections used sklearn on CPU as the running example. GPU-based deep learning models introduce additional complexity around device selection, memory management, and containerization. BentoML handles GPU serving through the `resources` and environment configuration.

### Defining a GPU service class

```python
# gpu_service.py
import bentoml
import torch
from transformers import pipeline
from pydantic import BaseModel
from typing import List

class SentimentInput(BaseModel):
    text: str
    max_length: int = 512

class SentimentOutput(BaseModel):
    label: str
    score: float

@bentoml.service(
    resources={
        "gpu": 1,                    # request exactly 1 GPU
        "gpu_type": "nvidia-t4",     # optional: constrain to T4s
        "memory": "8Gi",
        "cpu": "2",
    },
    traffic={
        "timeout": 60,
        "max_concurrency": 32,
    },
)
class SentimentClassifier:
    # BentoML ensures this class is instantiated inside a process
    # that has exactly one GPU visible (CUDA_VISIBLE_DEVICES=<assigned>)
    bento_model = bentoml.models.get("sentiment_clf:latest")

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = self.bento_model.path
        self.pipe = pipeline(
            "text-classification",
            model=model_path,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        torch.cuda.empty_cache()

    @bentoml.api(
        batchable=True,
        max_batch_size=16,
        max_latency_ms=100,
    )
    def predict(self, inputs: List[SentimentInput]) -> List[SentimentOutput]:
        texts = [inp.text for inp in inputs]
        results = self.pipe(
            texts,
            truncation=True,
            max_length=inputs[0].max_length,
            padding=True,
        )
        return [SentimentOutput(label=r["label"], score=r["score"]) for r in results]
```

The `resources={"gpu": 1}` annotation has two effects:
1. In Kubernetes deployments, it sets `limits: nvidia.com/gpu: 1` on the pod spec, ensuring the pod is scheduled on a GPU node.
2. In local `bentoml serve`, it validates that at least one CUDA-capable GPU is available before starting (failing fast rather than silently running on CPU).

### Multi-GPU serving with `--production`

For a service that should run one model copy per GPU:

```bash
# Serve on 4 GPUs — each worker gets one GPU
# BentoML assigns CUDA_VISIBLE_DEVICES automatically
BENTOML_NUM_WORKERS=4 bentoml serve gpu_service:SentimentClassifier --production

# Equivalent Docker invocation (the container sees all 4 GPUs via --gpus all;
# BentoML distributes them across workers)
docker run --rm --gpus all -p 3000:3000 sentiment_clf:latest \
    bentoml serve sentiment_clf:latest --production --workers 4
```

When you have a model too large for a single GPU — such as a 13B parameter language model on T4 (16GB) GPUs — you need tensor parallelism, which BentoML handles through `@bentoml.depends` with a DeepSpeed or Accelerate-based runner in a separate service class. This is more complex than native Triton or vLLM handling of multi-GPU, which is one reason large LLMs typically use vLLM or TGI rather than BentoML for serving. BentoML's GPU story is strongest for single-GPU models.

### The bentofile.yaml for GPU models

```yaml
# bentofile.yaml for GPU sentiment classifier
service: "gpu_service:SentimentClassifier"

include:
  - "*.py"
  - "config/*.yaml"

python:
  packages:
    - torch>=2.2.0
    - transformers>=4.38.0
    - accelerate>=0.27.0
    - pydantic>=2.0.0

docker:
  base_image: "nvcr.io/nvidia/pytorch:24.02-py3"   # NVIDIA's pre-built PyTorch image
  cuda_version: "12.3.2"
  system_packages:
    - libgomp1
  env:
    PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb=512"
    TOKENIZERS_PARALLELISM: "false"
```

Using `nvcr.io/nvidia/pytorch:24.02-py3` as the base image instead of the slim Python image gives you a pre-installed CUDA toolkit, cuDNN, NCCL, and a recent PyTorch build with all CUDA extensions compiled. This cuts 15–30 minutes off your image build time and eliminates a class of CUDA version mismatch bugs.

#### Worked example: T4 GPU latency and cost optimization

You are deploying a DistilBERT sentiment classifier on a single T4 GPU (16GB VRAM, available as an `n1-standard-4` + T4 on GCP for approximately \$0.35/hour). Your SLO is p99 < 200ms, and your expected load is 200 req/s peak.

Model characteristics: DistilBERT (66M params, ~250MB FP32, ~125MB FP16). A single inference with batch_size=1: 8ms on T4 with FP16. With batch_size=16: 14ms total = 0.875ms per item.

With BentoML adaptive batching at `max_batch_size=16, max_latency_ms=50`:
- Worst-case p99: 50ms (batch window) + 14ms (inference) = 64ms. Well within the 200ms SLO.
- At 200 req/s: batch windows fill in 200 / (1000/50) = 10 items on average. Batches fire at ~10 items, giving average batch efficiency of 62.5%.
- Throughput capacity: at full 16-item batches firing every 14ms: 16/0.014 = 1,143 req/s. Single T4 can handle 5.7x your peak load with headroom.

At \$0.35/hour for the T4 instance, you are paying \$0.35 / 1143 req/s = \$0.000087 per 1,000 requests at maximum efficiency. At your 200 req/s operating point (17.5% of peak), the effective cost per 1,000 requests including idle time is \$0.35/hour / (200 * 3600) = \$0.000486 per 1,000 requests. Deploying BentoML adaptive batching versus a naive sequential server (limited to roughly 125 req/s at 8ms per request with 100ms overhead) saves approximately 37% in compute cost at your target load.


## MLServer multi-model serving and memory management

One operationally important MLServer feature is its ability to serve multiple models from a single process with shared memory. Instead of running one MLServer process per model — which wastes memory on per-process overhead and slows down deployment management — you can serve many models from a single MLServer instance.

### Multi-model directory layout

```
models/
├── settings.json                    # global MLServer settings
├── adult_income_clf/
│   ├── model-settings.json
│   └── adult_income_clf.joblib
├── churn_predictor/
│   ├── model-settings.json
│   └── churn_predictor.json          # XGBoost JSON format
└── sentiment_clf/
    ├── model-settings.json
    └── (HuggingFace model files)
```

The `settings.json` at the root of the models directory controls global MLServer behavior:

```json
{
  "debug": false,
  "host": "0.0.0.0",
  "http_port": 8080,
  "grpc_port": 8081,
  "metrics_port": 8082,
  "workers": 4,
  "max_buffer_size": 104857600,
  "cors_settings": {
    "allow_origins": ["*"],
    "allow_methods": ["*"],
    "allow_headers": ["*"]
  }
}
```

With this layout, a single `mlserver start ./models/` command serves all three models. Each model gets its own V2 endpoint under `/v2/models/{model_name}/`. The three models share a single Python process and the same worker pool, which reduces memory overhead significantly compared to three separate processes.

The trade-off: if one model has a bug that crashes the Python runtime, all models in the same process crash together. For production environments with strict isolation requirements, run separate MLServer processes per model or per model family, and use Kubernetes pod anti-affinity rules to place critical models on separate nodes.

### Model lazy loading and warm-up

MLServer supports lazy loading: you can configure a model to not load on startup but instead load on the first request. This is useful for fleets with many infrequently-used models where loading all models at startup would exceed memory limits.

```json
{
  "name": "rarely_used_model",
  "implementation": "mlserver_sklearn.SKLearnModel",
  "parameters": {
    "uri": ".",
    "version": "v1"
  },
  "warm_up": false,
  "max_buffer_size": 52428800
}
```

When `warm_up: false`, the model is loaded on the first inference request, not at server startup. The first request's latency absorbs the model load time (3–8 seconds for sklearn, 15–30 seconds for BERT). For models with strict p99 SLOs, ensure `warm_up: true` (the default) and set your `minReplicas` accordingly.

### Loading models from remote storage

In Kubernetes environments, models live in S3, GCS, or Azure Blob Storage rather than on local disk. MLServer can load models directly from these URIs in the `model-settings.json`:

```json
{
  "name": "adult_income_clf",
  "implementation": "mlserver_sklearn.SKLearnModel",
  "parameters": {
    "uri": "s3://ml-models/adult-income-clf/v2/adult_income_clf.joblib",
    "version": "v2.0.0"
  }
}
```

When using KServe, model download is handled by the KServe storage initializer init container before MLServer starts — so the `uri` in KServe's `InferenceService` spec points to the S3 path, but MLServer itself is given a local filesystem path by the time it runs. This separation means MLServer does not need cloud storage credentials in the serving container.


## Troubleshooting common failure modes

Both BentoML and MLServer have characteristic failure patterns that are worth knowing before you hit them in production at 2 AM.

### BentoML: "Runner is not ready" on startup

In BentoML 0.13.x (the Runner pattern), a common failure is the service starting before the Runner subprocess has fully initialized. The symptom is HTTP 503 responses on the first few requests after startup. The fix:

```bash
# Check BentoML startup logs for runner initialization
docker logs <container_id> 2>&1 | grep -E "runner|ready|error"

# In BentoML 1.x, this is less common because the service class
# loads the model in __init__ and the readiness probe only passes
# after __init__ completes successfully.
```

In BentoML 1.x, if your service fails to load the model (`__init__` raises an exception), the process exits and Kubernetes restarts the pod. Watch for `CrashLoopBackOff` with `kubectl get pods` and inspect with `kubectl logs <pod-name> --previous`.

### BentoML: OOM during model loading

If your model is larger than the container's memory limit, the pod is OOM-killed before it can serve any requests. This shows up as the container never reaching the Ready state (`kubectl describe pod` shows `OOMKilled`).

```bash
# Check pod events for OOM
kubectl describe pod income-classifier-abc123 -n ml-serving | grep -A5 "OOMKilled"

# Increase memory limits in bentofile.yaml
# resources:
#   memory: "8Gi"   # increase from 2Gi
```

For large models, add a memory headroom multiplier: if your model is 4 GB in memory, set the container memory limit to 4 GB * 1.5 = 6 GB to account for Python overhead, OS buffers, and batch memory allocation during inference.

### MLServer: model not found at startup

The most common MLServer startup failure is a misconfigured `model-settings.json` where the `uri` does not resolve to an actual file. MLServer logs the error but starts the HTTP server anyway (for health probes), so the pod looks ready but returns 404 for inference requests.

```bash
# Check MLServer logs for model loading errors
kubectl logs mlserver-pod-abc123 -n ml-serving | grep -E "ERROR|model|load"
# Output: ERROR - Could not load model: file not found at ./adult_income_clf.joblib

# Verify the model file is present in the pod
kubectl exec -it mlserver-pod-abc123 -n ml-serving -- ls -la /mnt/models/
```

If using KServe's storage initializer, check the init container logs:

```bash
kubectl logs mlserver-pod-abc123 -c storage-initializer -n ml-serving
# Output: Downloading model from s3://ml-models/adult-income-clf/v2/
# Error: AccessDenied: s3://ml-models/adult-income-clf/v2/
```

### Latency regression after a model update

If p99 latency spikes after updating a model but error rates remain low, the most likely cause is the new model being larger or slower than the previous version. Check:

1. **Model size**: `kubectl exec -it mlserver-pod -- du -sh /mnt/models/` — a larger model takes longer to load and may have different inference time characteristics.
2. **Batch size distribution**: for BentoML, check the `bentoml_service_batch_size` histogram. If the p50 batch size dropped, fewer requests are being batched together (possible if the new model is slower and the batch window fills slower).
3. **CPU/GPU utilization**: `kubectl top pods` — a higher utilization without higher throughput indicates the new model is using more compute per inference.
4. **Memory bandwidth**: for GPU models, use `nvidia-smi` inside the pod to check if the new model is running into HBM bandwidth limitations.


## When to use this (and when not to)

### Use BentoML when:

- Your team is Python-first and will not or cannot deploy Kubernetes with KServe/Seldon.
- Your model endpoint needs custom preprocessing, business logic, or feature store integration that is complex enough to be painful in YAML.
- You need adaptive batching for CPU-bound classical ML models at throughput > 500 req/s.
- You want a self-contained, portable OCI image that does not depend on any serving platform infrastructure.
- Your data science and ML engineering teams want to own the full deployment lifecycle without depending on a platform team.
- You are doing rapid prototyping and need to iterate on the model and serving code simultaneously with hot reload.

### Do not use BentoML when:

- Your organization already has KServe or Seldon Core as the standard serving platform and adding a second serving system creates more maintenance overhead than it saves.
- Your models need to participate in V2 protocol pipelines (Seldon multi-step pipelines, KServe transformers, Triton ensemble with MLServer steps).
- Your primary latency target is single-request p99 below 5ms — MLServer's gRPC interface is consistently faster for single-request workloads.
- You need Alibi Explain integration for per-prediction explanations in production.

### Use MLServer when:

- You are already running KServe or Seldon Core and want to add models without new infrastructure.
- You need V2 protocol compliance so your models are interchangeable with Triton-served models in the same pipeline.
- Your model is a standard framework (sklearn, XGBoost, LightGBM, HuggingFace) and you want zero custom code in the serving layer.
- Your organization wants unified monitoring and client libraries across a heterogeneous model fleet.
- You need KServe's native canary rollouts, blue-green deployments, and traffic splitting without writing Helm charts or custom Kubernetes controllers.

### Do not use MLServer when:

- You have complex preprocessing that does not fit into MLServer's runtime model and you would need a custom runtime that is harder to maintain than a BentoML service class.
- You are serving outside Kubernetes and do not want to run the full KServe/Istio/Knative stack on a VM.
- Your team has no Kubernetes expertise and the operational complexity of KServe is higher than the team can safely manage.
- Your model requires adaptive batching with very tight latency SLOs (< 10ms) where Python-layer batching adds too much overhead — in this case, consider Triton's native batching instead.


## Migrating between BentoML and MLServer

As teams grow, the common migration trajectory is BentoML → MLServer + KServe. Understanding the pain points of this migration lets you either defer it longer or plan it more efficiently.

The primary challenge in migration is not the model itself — saving a model as joblib and writing `model-settings.json` is trivial. The challenge is the preprocessing and postprocessing logic that has accumulated in the BentoML service class: feature engineering, input validation, output formatting, database lookups, feature store calls, audit logging. Each of these must be extracted into a separate component that can run as a Kubernetes-native service.

A pragmatic migration strategy is to wrap your BentoML service as a Seldon Custom Server (a KServe-compatible container that implements V2 protocol endpoints), keeping all the Python business logic intact while gaining KServe's orchestration. The BentoML container exposes its own REST API; you write a thin V2 adapter sidecar that translates V2 requests to BentoML's native API. This hybrid runs in production while you incrementally refactor the business logic into proper MLServer custom runtimes.

For teams going the other direction — adopting BentoML after experience with KServe — the transition is usually driven by a new product team that needs fast iteration without platform team involvement. BentoML's `bentoml import` command can import a model from a KServe-managed S3 URI into the local model store:

```bash
# Import a model artifact from S3 into BentoML model store
bentoml models import adult_income_clf \
    --uri s3://ml-models/adult-income-clf/v2/adult_income_clf.joblib \
    --framework sklearn
```

This gives the new team a local development copy of the model with full BentoML tooling without touching the production KServe deployment.


## Key takeaways

1. **BentoML and MLServer solve the same problem with opposing philosophies**: BentoML is Python-first and developer-local; MLServer is protocol-first and Kubernetes-native. Neither is universally superior — team topology and existing infrastructure determine the right choice.

2. **BentoML adaptive batching delivers 3–8x throughput improvement** for CPU-bound classical ML models with a minimal configuration change (`batchable=True`, `max_batch_size`, `max_latency_ms`). For models where inference time is much less than `max_latency_ms`, this is nearly free throughput gain.

3. **The V2 inference protocol is the lingua franca of Kubernetes-native model serving.** A client written to V2 runs against Triton, MLServer, KServe, and TorchServe with zero code changes. If you serve a heterogeneous model fleet, V2 compliance eliminates entire categories of client library maintenance overhead.

4. **MLServer's packaging speed advantage is real but conditional.** Pointing a stock MLServer Docker image at a `model-settings.json` is faster than `bentoml containerize`, but it requires a running MLServer or KServe environment. BentoML's OCI image runs anywhere Docker runs, with no external dependencies.

5. **KServe canary rollouts require no application code.** Traffic percentages are set in YAML and enforced by Istio. A 10% canary to 100% promotion is three `kubectl` commands, not a code change.

6. **Cold start determines your minimum replica count.** MLServer with a 4 MB sklearn model: 4–8 second cold start. MLServer with a 400 MB BERT model: 20–35 seconds. Never set `minReplicas: 0` if your p99 SLA is shorter than the cold-start time for your model class.

7. **BentoML's `@bentoml.depends` composes services for multi-model pipelines.** For simple N-step compositions, this is more ergonomic than writing Kubernetes inter-service networking. For complex DAG-shaped pipelines, Seldon Core's V2-based orchestration is more appropriate.

8. **Observability is not optional.** The Prometheus batch size histogram (`bentoml_service_batch_size`) tells you whether your batching configuration is appropriate for your actual traffic pattern. The MLServer `mlserver_model_loaded` gauge catches model loading failures that the Kubernetes readiness probe may not surface immediately.

9. **Preprocessing complexity is the primary migration friction.** If your BentoML service class contains more than ~50 lines of preprocessing and business logic, plan the MLServer migration around that logic — not the model artifact itself.

10. **For the first production model in a Python-native team, BentoML is almost always the right choice.** For the 20th model on a platform team's Kubernetes cluster with KServe already installed, MLServer is almost always the right choice. Avoid the trap of choosing based on technical capability rather than team fit — both tools are technically adequate for most use cases.


## Further reading

- [BentoML official documentation](https://docs.bentoml.com) — The definitive reference for BentoML 1.3+ service definition, bentofile.yaml syntax, model store API, and cloud deployment guides. The "Quickstart" section is an accurate representation of the real workflow.
- [MLServer documentation](https://mlserver.readthedocs.io) — Covers built-in runtimes, custom runtime development, and V2 protocol reference. The "Runtimes" section documents every `implementation` value and its expected model format.
- [Open Inference Protocol specification](https://kserve.github.io/website/latest/modelserving/v1beta1/serving_runtime/) — The formal CNCF specification for the V2 HTTP/REST and gRPC inference protocols. Essential reading before writing a custom V2-compatible server.
- [KServe InferenceService documentation](https://kserve.github.io/website/latest/modelserving/v1beta1/serving_runtime/) — Official KServe docs for the `InferenceService` CRD, storage URIs, autoscaling configuration, and canary rollout patterns.
- [Triton Inference Server deep dive](/blog/machine-learning/model-serving/triton-inference-server-deep-dive) — The previous post in this series covers Triton's V2 implementation, dynamic batching, and ensemble pipelines in depth. Triton's batching subsystem is the production standard against which BentoML's adaptive batching should be compared for GPU workloads.
- [Ray Serve deep dive](/blog/machine-learning/model-serving/ray-serve-deep-dive) — Ray Serve's deployment model and autoscaling patterns. Useful for understanding where Ray Serve fits relative to BentoML for Python-native serving.
- [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — The series introduction: the SLO triangle, Little's Law, and why the serving layer is a distinct engineering discipline from model training.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — The series capstone: a complete decision tree from model type and team context to recommended serving stack.
