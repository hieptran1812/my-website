---
title: "Ray Serve deep dive: deployments, pipelines, and autoscaling for Python-first inference"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master Ray Serve's deployment model, DeploymentHandle composition, autoscaling, batching, and fault tolerance to build production-grade multi-model serving pipelines in pure Python."
tags:
  [
    "model-serving",
    "inference",
    "ray-serve",
    "ml-infrastructure",
    "autoscaling",
    "deep-learning",
    "distributed-systems",
    "python",
    "mlops",
    "rag",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 49
image: "/imgs/blogs/ray-serve-deep-dive-1.png"
---

There is a class of model serving problem that Triton Inference Server cannot solve and vLLM cannot solve and TorchServe cannot solve — not because they are bad tools, but because the bottleneck is not a single model. It is the Python code that glues models together.

Imagine you are building a production RAG pipeline: a query arrives, an embedding model turns it into a vector, a FAISS index retrieves the top-20 documents, a cross-encoder reranker narrows those to five, and a language model generates the final answer while streaming tokens back to the user. Each stage has different hardware requirements, different throughput capacity, different scaling curves. The embedding model is CPU-bound and blazes at 2,000 calls/second on a single core. The generator bottlenecks at 40 requests/second per A100. Triton can batch the generator, but it cannot express "this deployment needs four GPU replicas and that one needs two CPU replicas and they talk to each other." TorchServe can serve a single model. vLLM is a masterpiece for LLM inference but it is fundamentally a one-model server. None of them natively express "run this Python code on those machines, let me compose them into a graph, and autoscale each independently."

Ray Serve does. It is a scalable, framework-agnostic serving library built on top of Ray's distributed actor model. The model is simple: a **Deployment** is a named pool of Python actor replicas. You compose Deployments into graphs with `serve.bind()`. You autoscale each Deployment independently. The HTTP proxy and internal router handle the rest. The philosophy is Python-first — you write a Python class, add two decorators, and you have a production-grade microservice.

The diagram below is the mental model: an HTTP Proxy on port 8000 receives your request, hands it to Serve's internal router, and the router dispatches to one of the named deployment pools. Each pool lives on Ray worker nodes, which can carry GPU, CPU, or any mix of resources. The EmbeddingDeploy gets two CPU replicas; the GenerationDeploy gets four GPU replicas on A100s. Each scales independently.

![Ray Serve architecture: HTTP Proxy routes requests through the internal router to named deployment replica pools, each running on Ray worker nodes](/imgs/blogs/ray-serve-deep-dive-1.webp)

By the end of this post you will be able to: declare and compose Ray Serve deployments from scratch; wire multi-model pipelines with `DeploymentHandle`; configure the autoscaler's queue-depth controller with the right delay parameters; accelerate throughput with `@serve.batch`; allocate fractional GPU resources for high-density deployments; respond to replica crashes with proper health-check and restart configuration; and read the Prometheus metrics that expose Serve's internal state. We will build our intuition against a running RAG pipeline example that grows from a single deployment into a full seven-stage serving graph.

This post is part of the [Model Deployment and Serving series](/blog/machine-learning/model-serving/what-is-model-serving). If you are new to the SLO triangle (latency, throughput, cost) or the basics of batching and queueing theory, read the [batching fundamentals post](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) first. The concepts there — Little's Law, batch formation latency budgets, max-batch-size tuning — all apply directly to `@serve.batch` configuration. If your use case is primarily LLM serving without pipeline composition, the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) is more directly applicable.

## 1. The Ray execution model: actors, tasks, and the object store

Before Serve makes sense, Ray itself needs to make sense. Ray is a framework for distributed Python computation. It has three primitives: **remote tasks** (stateless Python functions), **remote actors** (stateful Python classes), and the **distributed object store** (shared memory for passing data between workers without serialization round-trips).

A Ray cluster is a head node plus worker nodes. The head node runs the Global Control Store (GCS) — Ray's equivalent of a coordinator. Worker nodes run an agent process and accept actor placement from the scheduler. When you call `ray.init()`, you connect to a running cluster (or start a local one).

Ray Serve is implemented as a set of Ray actors. The core components are:
- **HTTP Proxy actor**: one per node in the cluster, receives HTTP requests and fan them out to the router
- **Router (controller) actor**: runs on the head node, tracks which replicas are alive and load-balances across them
- **Deployment replica actors**: your code, running in their own Python processes, one per replica

When you call `serve.run()`, the controller actor spawns the configured number of replica actors across the cluster. When a request arrives at the HTTP proxy, the proxy talks to the controller to find a live replica, then dispatches the request directly to that replica's mailbox. The replica processes it and returns the result through Ray's object store reference passing.

This architecture has important implications for latency and throughput:
- Each replica is isolated — one replica's GC pause or CPU spike does not block others
- Replicas can hold large model weights without sharing them across Python process boundaries (no GIL contention on model weights)
- Scaling up adds actors; scaling down removes them; neither operation touches code or configuration on surviving replicas

### The request lifecycle in detail

Understanding the request lifecycle helps you reason about where latency comes from and what to optimize:

1. **Client sends HTTP POST** to `http://<serve-host>:8000/rag` (or whatever `route_prefix` you configured).
2. **HTTP Proxy actor receives the request.** The proxy is a Starlette ASGI server running inside a Ray actor. It parses the HTTP request and issues a call to the Router actor to get an available replica handle.
3. **Router selects a replica.** The default policy is power-of-two choices: pick two replicas at random, route to the one with fewer in-flight requests. For `num_replicas=1`, there is no choice to make. For `num_replicas=4`, this policy achieves near-optimal load balancing under bursty traffic without the overhead of a centralized least-connections counter.
4. **Proxy dispatches request to the selected replica.** This is a Ray actor method call — the request object is placed in the replica's async event loop mailbox.
5. **Replica's `__call__` coroutine runs.** The coroutine can yield (via `await`) to process other requests while waiting for I/O or GPU compute. A single-threaded event loop inside the replica processes requests in FIFO order unless you configure `asyncio_max_concurrent_queries` to allow concurrency.
6. **Response is returned via Ray's object store.** The result travels back through the object store reference to the HTTP proxy, which serializes it as an HTTP response.

The latency breakdown for a typical non-batched request:
- HTTP parsing + proxy overhead: 0.3–0.8 ms
- Router round-trip (select replica): 0.1–0.3 ms
- Actor dispatch + queue wait: 0.2–0.4 ms
- Model forward pass: your model's inference time
- Response serialization + transmission: 0.2–0.5 ms

For a 180 ms generation request, the overhead is < 2 ms — negligible. For a 3 ms embedding request, the overhead is 60–70% of the total latency. This is why Ray Serve is better suited for compute-heavy workloads than for sub-5 ms latency requirements.

### How Ray Serve compares to a raw FastAPI + Gunicorn stack at the infrastructure level

To make the overhead concrete: a raw Gunicorn + FastAPI stack serving a BERT embedding model on a single A100 node achieves ~420 req/s at batch size 16 with a 15 ms p50 latency. The same model wrapped in Ray Serve achieves ~380 req/s at batch size 16 with a 17 ms p50 latency — roughly 10% throughput overhead and 2 ms latency overhead. This overhead comes from the actor dispatch cost and the async event loop context switch.

The trade-off is clear: for a single model on a single node, raw FastAPI is faster. But the moment you need dynamic scaling, rolling updates, multi-model composition, or multi-node distribution, maintaining a raw FastAPI stack requires custom infrastructure code that Ray Serve provides out of the box. The 10% overhead is the price of the infrastructure layer you no longer have to write yourself. For most production workloads, that is an excellent exchange.

## 2. Core concepts: deployments, the decorator, and serve.run()

A **Deployment** wraps a Python callable (a function or a class) with a pool of Ray actor replicas. You declare it with the `@serve.deployment` decorator. When you call `serve.run()`, Serve launches the requested number of actor replicas across your cluster and hooks them up to the HTTP proxy. The decorator does not start anything — it registers a recipe.

```python
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    max_concurrent_queries=100,   # backpressure: queue limit per replica
)
class EmbeddingModel:
    def __init__(self):
        # __init__ runs once per replica at startup time, not per request
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Embedding model loaded on {ray.get_runtime_context().get_node_id()}")

    async def encode(self, texts: list[str]) -> list[list[float]]:
        # Async method: event loop yields to other requests while model runs
        return self.model.encode(texts).tolist()

    async def __call__(self, request):
        data = await request.json()
        return {"embeddings": await self.encode(data["texts"])}

# Wire the application (no replicas started yet)
app = EmbeddingModel.bind()
# Start replicas and expose at /embed
serve.run(app, name="embedding_service", route_prefix="/embed")
```

The `@serve.deployment` decorator accepts three important categories of configuration:
- **Replica count / autoscaling**: `num_replicas` (fixed), or `autoscaling_config` (dynamic)
- **Resource allocation**: `ray_actor_options` — cpu, gpu, memory, accelerator type
- **Request handling**: `max_concurrent_queries` (queue depth per replica), `graceful_shutdown_timeout_s`

The architectural stack that a Serve deployment sits in:

![Ray Serve deployment anatomy: nested stack from Ray Cluster at the outermost layer down to HTTP Endpoint at the innermost, with @serve.deployment and serve.bind() as the key composition layers](/imgs/blogs/ray-serve-deep-dive-2.webp)

### How Serve differs from a plain FastAPI app

When you run a FastAPI app on a VM, you have one Python process. If you want two processes, you run Gunicorn with two workers. If you want GPU isolation between them, you manage `CUDA_VISIBLE_DEVICES` by hand. If you want to scale one endpoint differently from another, you need separate pods, separate Kubernetes Services, separate HPAs, and separate log streams to correlate.

With Ray Serve, the cluster is the unit. Any replica can run on any node. You express resource requirements in `ray_actor_options`, and Ray's scheduler places replicas wherever those resources exist. Scaling a deployment changes the desired replica count; Serve adds or removes actors without redeploying anything else in the application. Two deployments in the same application can have completely different resource profiles and scale curves.

The other key difference is **`__init__` semantics**. A Ray actor's `__init__` runs once, when the actor launches. Your model weights load once per replica, not once per request. A FastAPI + Gunicorn worker also loads the model once per process, but managing the relationship between Gunicorn workers, GPU memory, and dynamic scaling requires manual CUDA device assignment. In Ray Serve, you declare resource requirements declaratively and the scheduler handles placement.

### `serve.bind()` for composition

`serve.bind()` creates a **DeploymentNode** — a description of how one deployment passes its output as an argument to another. You chain these to form a directed acyclic graph of compute:

```python
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
class Retriever:
    def __init__(self, embedder_handle: DeploymentHandle):
        import faiss, numpy as np
        self._embedder = embedder_handle
        self._index = faiss.read_index("corpus.faiss")
        self._docs = open("docs.jsonl").readlines()

    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # Call the embedder via handle — this is an actor call, not HTTP
        emb_ref = await self._embedder.encode.remote([query])
        embs = await emb_ref
        vec = np.array(embs[0], dtype="float32").reshape(1, -1)
        _, indices = self._index.search(vec, top_k)
        return [self._docs[i] for i in indices[0]]

@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 8,
                        "target_num_ongoing_requests_per_replica": 5},
)
class Generator:
    def __init__(self, retriever_handle: DeploymentHandle):
        from vllm import LLM, SamplingParams
        self._retriever = retriever_handle
        self.llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
                       tensor_parallel_size=1, gpu_memory_utilization=0.90)
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    async def __call__(self, request):
        data = await request.json()
        query = data["query"]
        docs = await self._retriever.retrieve.remote(query)
        context = "\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        outputs = self.llm.generate([prompt], self.sampling_params)
        return {"answer": outputs[0].outputs[0].text}

# Build the computation graph
embedder = EmbeddingModel.bind()
retriever = Retriever.bind(embedder_handle=embedder)
generator = Generator.bind(retriever_handle=retriever)

# Launch the full graph from a single call
serve.run(generator, name="rag_pipeline", route_prefix="/rag")
```

`serve.bind()` does not instantiate anything. It creates a bind record that `serve.run()` uses to topologically sort and launch deployments in dependency order. Circular dependencies are caught at bind time. When `serve.run()` is called with the `generator` node, it inspects the bind tree, finds `embedder` and `retriever` as upstream dependencies, and starts their replicas first.

### Deployment versioning and rolling updates

One of Ray Serve's most operationally valuable features is zero-downtime rolling updates. When you modify a deployment's code or config and call `serve.run()` again, Serve performs a rolling update:

1. New replicas are launched with the updated code.
2. Once new replicas pass their first health check, the router starts sending new requests to them.
3. Old replicas continue processing their in-flight requests.
4. After `graceful_shutdown_timeout_s` (default 20 s), old replicas are removed.

This means you never drop requests during a code push. The only exception is if the new code breaks initialization (`__init__` throws an exception) — in that case, Serve keeps the old replicas running and marks the new deployment as failed.

```python
# Update a deployment: just change the code and call serve.run() again
@serve.deployment(
    num_replicas=3,
    graceful_shutdown_timeout_s=30,   # old replicas get 30s to finish
    graceful_shutdown_wait_loop_s=2,  # check every 2s if in-flight requests drained
)
class EmbeddingModel:
    # v2: switched to a better model
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-mpnet-base-v2")  # was all-MiniLM-L6-v2

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    async def __call__(self, request):
        data = await request.json()
        return {"embeddings": await self.encode(data["texts"])}

# Re-running this with the updated class definition performs a rolling update
serve.run(EmbeddingModel.bind(), name="embedding_service", route_prefix="/embed")
```

Rolling updates work across the entire graph: when you update one deployment in a multi-deployment graph, only that deployment's replicas are replaced. The rest of the graph continues serving from its existing replicas.

### Deployment configuration inheritance with `serve.config`

For production deployments, you typically want to separate code from configuration. Ray Serve supports YAML-based serve configs that override the defaults in your decorators:

```yaml
# serve_config.yaml
applications:
  - name: rag_pipeline
    route_prefix: /rag
    import_path: myapp.deployments:build_rag_app
    deployments:
      - name: EmbeddingModel
        num_replicas: 2
        ray_actor_options:
          num_cpus: 1
      - name: GeneratorDeploy
        autoscaling_config:
          min_replicas: 1
          max_replicas: 8
          target_num_ongoing_requests_per_replica: 5
        ray_actor_options:
          num_gpus: 1
```

```bash
# Deploy from config (good for GitOps workflows)
serve deploy serve_config.yaml
# Check status
serve status
# List all applications
serve list
```

The config file approach separates cluster-specific parameters (replica counts, resource allocation) from model code. Data scientists write the model class; MLOps engineers tune the serving config. Both files live in version control; a CI/CD pipeline deploys them independently.

## 3. DeploymentHandle: calling one deployment from another

Inside a Serve deployment, the right way to call a sibling deployment is through its `DeploymentHandle`. This is not HTTP — it is a direct Ray actor method call, routed through the cluster scheduler with ~200–400 µs latency overhead instead of 1–5 ms TCP round-trip.

```python
from ray.serve.handle import DeploymentHandle

@serve.deployment(num_replicas=1)
class RouterDeploy:
    def __init__(
        self,
        small_model: DeploymentHandle,
        large_model: DeploymentHandle,
    ):
        self._small = small_model
        self._large = large_model

    async def __call__(self, request):
        data = await request.json()
        text = data["text"]
        # Route short inputs to the small model, long inputs to the large
        if len(text.split()) < 50:
            ref = await self._small.generate.remote(text)
        else:
            ref = await self._large.generate.remote(text)
        result = await ref
        return {"output": result}
```

`await handle.method.remote(args)` returns an `ObjectRef` — a future pointing at the result in Ray's distributed object store. You await it to block until the result is ready. You can also fan out multiple calls and gather them concurrently:

```python
async def dual_score(self, text: str):
    # Fan out: both model calls start simultaneously
    ref_a = await self.model_a.score.remote(text)
    ref_b = await self.model_b.score.remote(text)
    # Await both — total wait is max(latency_a, latency_b), not sum
    score_a, score_b = await asyncio.gather(ref_a, ref_b)
    return {"model_a": score_a, "model_b": score_b}
```

The handle respects the **load balancing policy** of the target deployment. By default, Ray Serve uses a power-of-two random choice: pick two random live replicas, send to the one with fewer in-flight requests. Under bursty traffic patterns, this consistently outperforms round-robin, which can pile requests on a slow replica while fast ones sit idle.

### A/B testing via probability-based routing

`DeploymentHandle` makes A/B testing a first-class concern. You hold handles to both the control and treatment models and route by a configurable coin flip:

```python
import random

@serve.deployment(num_replicas=1)
class ABRouter:
    def __init__(
        self,
        control: DeploymentHandle,
        treatment: DeploymentHandle,
        treatment_fraction: float = 0.1,
    ):
        self._control = control
        self._treatment = treatment
        self._frac = treatment_fraction

    async def __call__(self, request):
        data = await request.json()
        if random.random() < self._frac:
            ref = await self._treatment.predict.remote(data["input"])
            variant = "treatment"
        else:
            ref = await self._control.predict.remote(data["input"])
            variant = "control"
        result = await ref
        return {"output": result, "variant": variant}
```

You can extend this with header-based routing (`request.headers["X-Model-Version"]`), query-parameter routing (`request.query_params.get("model", "default")`), or a dynamic policy loaded from a remote config store like Redis. None of this requires changing the deployment topology — you update the router's logic, reload the router deployment, and the pipeline keeps serving.

### Query parameter and header-based routing

For canary deployments, the router needs to direct specific traffic segments to specific model versions:

```python
@serve.deployment(num_replicas=1)
class CanaryRouter:
    def __init__(self, stable: DeploymentHandle, canary: DeploymentHandle):
        self._stable = stable
        self._canary = canary

    async def __call__(self, request):
        # Route beta users or explicit version requests to canary
        model_ver = request.headers.get("X-Model-Version", "stable")
        user_tier = request.headers.get("X-User-Tier", "regular")

        if model_ver == "canary" or user_tier == "beta":
            ref = await self._canary.generate.remote(await request.body())
            version_label = "canary"
        else:
            ref = await self._stable.generate.remote(await request.body())
            version_label = "stable"

        result = await ref
        return {"output": result, "model_version": version_label}
```

This is precisely the pattern used for champion-challenger model evaluation in production. The router logs which version served each request; you correlate with downstream outcome metrics to determine if the canary should be promoted or rolled back.

### Handle multiplexing and request deduplication

One advanced pattern is **handle multiplexing** — holding multiple handles to the same deployment and routing based on a routing key. This is useful when you have model shards that own different partitions of data:

```python
@serve.deployment(num_replicas=1)
class ShardedRetriever:
    def __init__(self, shard_handles: dict[str, DeploymentHandle]):
        self._shards = shard_handles  # {"shard0": handle, "shard1": handle, ...}
        self._num_shards = len(shard_handles)

    async def retrieve(self, query_vec, top_k: int = 5) -> list[str]:
        import asyncio, numpy as np
        # Broadcast to all shards simultaneously
        refs = [
            await self._shards[f"shard{i}"].search.remote(query_vec, top_k)
            for i in range(self._num_shards)
        ]
        # Await all shard results
        shard_results = await asyncio.gather(*refs)
        # Merge and re-rank
        all_docs = [doc for results in shard_results for doc in results]
        return sorted(all_docs, key=lambda d: d["score"], reverse=True)[:top_k]
```

For large-scale retrieval, this lets you distribute a FAISS index across multiple replicas (each holding a shard of the corpus), fan out a query to all shards simultaneously, and merge the results. The parallelism is `O(num_shards)` latency rather than `O(corpus_size)` serial latency.

## 4. The RAG pipeline: a running example with latency budgets

The running example is a Retrieval-Augmented Generation pipeline with seven stages, each deployed as an independent Ray Serve deployment.

![RAG pipeline as chained Ray Serve deployments: seven stages from EmbedDeploy to StreamingProxy, with per-stage replica counts and measured per-call latency](/imgs/blogs/ray-serve-deep-dive-3.webp)

Each stage has its own resource profile and scaling behavior:

| Deployment | Resources | Replicas | Per-call latency | Throughput cap |
|---|---|---|---|---|
| EmbedDeploy | 1 CPU | 2 | ~3 ms | ~600 req/s per replica |
| RetrieverDeploy | 1 CPU, 16 GB RAM | 2 | ~8 ms | ~200 req/s per replica |
| RerankerDeploy | 1 CPU | 1 | ~5 ms | ~300 req/s per replica |
| GeneratorDeploy | 1 A100 40GB | 4 | ~180 ms | ~5 req/s per replica |
| StreamingProxy | 0.2 CPU | 1 | ~5 ms overhead | ~500 req/s |

The pipeline's end-to-end p50 latency is dominated by the generator: `3 + 8 + 5 + 180 + 5 ≈ 200 ms`. The handle-call overhead between stages is ~400 µs each, adding ~2 ms total — negligible against a 200 ms generation budget.

The generator needs four replicas to hit 20 req/s aggregate throughput (`4 × 5 req/s = 20 req/s`). The embedder, retriever, and reranker combined can handle well over 100 req/s each with one or two replicas. The bottleneck is always the generator.

This is the core argument for Ray Serve over a monolithic serving framework: **per-stage replica counts are first-class configuration**. When the FAISS index grows and retrieval slows from 8 ms to 25 ms, you scale the RetrieverDeploy to 4 replicas without touching the generator configuration.

#### Worked example: latency budget allocation for a 500 ms SLO

Your SLO is p99 < 500 ms. You want to allocate budget per stage. The key constraint: the generator at batch size 1 takes 180 ms p50, but at p99 the variance can spike to 300 ms (due to input length variance and GPU thermal throttling). Let us allocate conservatively:

| Stage | p50 budget | p99 budget | Scaling lever |
|---|---|---|---|
| Embed | 5 ms | 10 ms | More CPU replicas |
| Retrieve | 15 ms | 30 ms | More FAISS shards, more replicas |
| Rerank | 8 ms | 15 ms | CPU replicas |
| Generate | 280 ms | 380 ms | More GPU replicas, batching |
| Network/proxy | 10 ms | 20 ms | Nginx tuning |
| **Total** | **318 ms** | **455 ms** | Within 500 ms p99 |

At this budget, the generator can use up to 380 ms. With four A100 replicas at 5 req/s each and continuous batching, you have 20 req/s throughput headroom — sufficient for a 300-user pilot. When you scale to 300 req/s, you need 60 GPU replicas or a faster model (a distilled 3B parameter model at 18 req/s per GPU would achieve the same throughput with fewer replicas and lower memory pressure).

## 5. Autoscaling: the queue-depth controller

Ray Serve's autoscaler is a **reactive queue-depth controller** — it does not predict future load. It measures current in-flight requests per deployment and reacts.

The decision rule: every second, the autoscaler checks whether the number of in-flight requests exceeds `target_num_ongoing_requests_per_replica × num_current_replicas`. If so, and the overload has persisted for at least `upscale_delay_s`, it adds replicas up to `max_replicas`. If load drops below target and has been low for `downscale_delay_s`, it removes replicas down to `min_replicas`.

![Autoscaling decision lifecycle: queue depth sampled, scale-up triggered, upscale_delay_s confirmation, new replica launched, model warm-up, replica ready for serving, then eventually downscale_delay_s before scale-down](/imgs/blogs/ray-serve-deep-dive-6.webp)

The desired replica count at time $t$:

$$N^*(t) = \left\lceil \frac{\text{in\_flight}(t)}{\text{target\_ongoing\_per\_replica}} \right\rceil$$

Clamped to $[\text{min\_replicas}, \text{max\_replicas}]$. The smoothing factors (`upscale_smoothing_factor`, `downscale_smoothing_factor`) apply fractional scaling to prevent a step-function response: with `upscale_smoothing_factor=0.5`, when the ideal count doubles, only half the gap is closed in one decision step.

```python
@serve.deployment(
    autoscaling_config={
        "min_replicas": 0,           # scale to zero when idle
        "max_replicas": 8,
        "initial_replicas": 2,       # start here before first traffic
        "target_num_ongoing_requests_per_replica": 5,
        "upscale_delay_s": 10,       # wait 10s before adding replicas
        "downscale_delay_s": 300,    # wait 5min before removing replicas
        "upscale_smoothing_factor": 0.5,    # gradual scale-up
        "downscale_smoothing_factor": 1.0,  # aggressive scale-down
        "look_back_period_s": 30,    # window for computing average load
    },
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=20,
)
class GeneratorDeploy:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
        )
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    async def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    async def __call__(self, request):
        data = await request.json()
        return {"answer": await self.generate(data["prompt"])}
```

### Scale-to-zero and cold starts

`min_replicas: 0` enables scale-to-zero — when no traffic flows, Serve removes all replicas, releasing GPUs. The first request after a quiet period triggers a cold start: Ray must launch a new actor, import the Python module, run `__init__` (which loads model weights), and execute any warm-up passes. For a 7B parameter model in FP16, weight loading from local NVMe SSD takes 15–45 seconds. NCCL initialization for multi-GPU models adds another 15–30 seconds.

![Autoscaling before-after: always-on with 2 replicas consuming 3.60/hr always vs scale-to-zero with 0 replicas at idle and 15-45s cold start for first request](/imgs/blogs/ray-serve-deep-dive-4.webp)

The decision of when to use scale-to-zero:

| Scenario | Scale-to-zero? | Reasoning |
|---|---|---|
| Dev / staging endpoint | Yes | Cold start acceptable; savings significant |
| Research eval harness | Yes | Nightly eval, human reviews results, 30s wait is fine |
| User-facing API, p99 < 500 ms | No | Cold start blows SLO |
| Batch processing endpoint | Yes | Caller expects async; cold start is invisible |
| Internal tool, low traffic | Yes, with keep-warm | Save overnight; prevent cold start during business hours |

The **keep-warm ping** pattern: schedule a lightweight HTTP request every 4 minutes to the endpoint. This keeps the autoscaler from removing the last replica. Cost: one idle replica's GPU time. Benefit: zero cold starts during business hours for a \$1.50/hour GPU — cost of the keep-warm is \$1.50/hr × 8 hrs = \$12/day vs \$0/day for scale-to-zero with cold starts. For non-interactive workloads, scale-to-zero with cold starts wins. For interactive workloads, the keep-warm ping wins.

#### Worked example: autoscaling cost model for a 7B chatbot

Scenario: Mistral 7B on AWS `g5.xlarge` (A10G 24GB, \$1.006/hr). Peak load 8 AM–10 PM (14 hrs), roughly 20 req/min. Overnight (10 hrs): near-zero traffic.

| Configuration | GPU-hours/day | Daily cost | Cold start risk |
|---|---|---|---|
| Always-on, 1 replica | 24 | \$24.14 | None |
| Always-on, 2 replicas (HA) | 48 | \$48.29 | None |
| Scale-to-zero, `downscale_delay_s=600` | ~16 | \$16.10 | Overnight |
| Scale-to-zero + keep-warm ping overnight | ~17 | \$17.10 | None |

Scale-to-zero with keep-warm costs \$7/day less than always-on HA while maintaining zero cold-start risk. At 100 endpoints, that is \$700/day saved — \$255,500/year. The autoscaler's parameters and cost model are worth understanding deeply.

### Autoscaling reactivity: tuning for bursty vs smooth traffic

The autoscaler has two failure modes: **over-eager scaling** (adds replicas on every traffic spike, wastes GPU-hours) and **under-responsive scaling** (does not add replicas fast enough, SLO breaches under sudden load). The parameters that control this trade-off:

**`upscale_delay_s`**: how long sustained overload must persist before new replicas are added. Default 30 s. For predictable workloads (daily traffic ramp at 9 AM), set to 60–120 s — the ramp is gradual and you can afford to be patient. For bursty workloads (viral social media content triggering sudden demand spikes), set to 5–10 s to respond faster.

**`downscale_delay_s`**: how long sustained underload must persist before replicas are removed. Default 600 s. The key consideration is the **cold start cost** — if you remove a replica and traffic returns 2 minutes later, the cold start adds 30+ seconds to user latency. Setting `downscale_delay_s=3600` (1 hour) prevents removing replicas during predictable daily traffic lulls (lunch dip, evening drop) at the cost of some GPU-hours.

**`target_num_ongoing_requests_per_replica`**: the steady-state utilization target. Setting this to 1 means "one in-flight request per replica" — effectively CPU-bound single-threaded processing. Setting it to 20 enables high concurrency per replica (appropriate when each request is I/O-bound rather than compute-bound, like a retrieval-only deployment that awaits a database call).

The practical tuning flow:
1. Load test your deployment to find the maximum sustainable QPS per replica before p99 latency exceeds your SLA.
2. Set `target_num_ongoing_requests_per_replica` to 70–80% of that maximum — leaving headroom for variance.
3. Set `upscale_delay_s` based on your traffic pattern's characteristic burst duration.
4. Set `downscale_delay_s` based on the cold start penalty vs idle GPU cost trade-off.

```python
# Example: tuned autoscaling for a production retrieval service
@serve.deployment(
    autoscaling_config={
        "min_replicas": 2,                                     # always 2 warm replicas
        "max_replicas": 16,
        "initial_replicas": 4,
        "target_num_ongoing_requests_per_replica": 8,          # from load test: 10 starts degrading
        "upscale_delay_s": 15,                                 # respond within 15s of sustained load
        "downscale_delay_s": 900,                              # keep replicas for 15min of idle
        "upscale_smoothing_factor": 0.8,                       # near-full response to scaling signal
        "downscale_smoothing_factor": 0.5,                     # gradual scale-down
    },
    max_concurrent_queries=32,
)
class RetrieverDeploy: ...
```

## 6. Resource allocation and fractional GPU

Every `@serve.deployment` accepts `ray_actor_options`, which maps directly to Ray's resource specification.

```python
@serve.deployment(
    num_replicas=4,
    ray_actor_options={
        "num_cpus": 0.5,          # share a core across 2 replicas
        "num_gpus": 0.5,          # fractional GPU: 2 replicas per GPU
        "memory": 4 * 1024**3,   # 4 GB RAM guarantee (soft limit)
    }
)
class SmallClassifier:
    def __init__(self):
        import torch
        # Two replicas share one physical GPU; each gets ~half VRAM
        self.model = torch.load("resnet50_classifier.pt").half().cuda()

    async def __call__(self, request):
        data = await request.json()
        with torch.no_grad():
            import torch
            logits = self.model(torch.tensor(data["input"]).cuda().unsqueeze(0))
        return {"label": int(logits.argmax()), "confidence": float(logits.softmax(-1).max())}
```

**Fractional GPU allocation** (`num_gpus: 0.5`) has two behaviors depending on hardware:

1. **Time-sliced sharing** (default on NVIDIA GPUs without MIG enabled): Ray schedules both replicas on the same physical GPU. They share VRAM and time-slice compute. Ray trusts the `num_gpus` fraction for scheduling purposes but does not enforce VRAM limits — a replica that declares `num_gpus: 0.5` can use 90% of VRAM if the other replica is idle.

2. **NVIDIA MIG (Multi-Instance GPU)**: on A100 and H100, you partition the GPU into isolated slices with hardware-guaranteed VRAM and compute isolation. Each MIG slice has its own L2 cache, DRAM bandwidth, and CUDA engine. A 7-way partition of an A100 80GB gives seven 10GB slices, each with 1/7 of the GPU's compute. Ray can be configured to treat each MIG instance as a separate GPU resource:

```bash
# Create 7 x 1g.10gb MIG instances on A100
sudo nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C

# Start Ray with MIG resources mapped
ray start --head --resources='{"MIG:1g.10gb": 7}'
```

```python
# Deploy to a MIG slice
@serve.deployment(
    ray_actor_options={"resources": {"MIG:1g.10gb": 1}},
)
class MIGEmbedder: ...
```

For BERT-sized models at FP16 (~440 MB), seven models fit in one A100 with full hardware isolation. At \$3.70/hr for a bare A100 node, that is \$0.53/hr per isolated model endpoint — comparable to CPU serving cost but with GPU compute for batched inference.

### Placement groups for gang scheduling

When a deployment requires multiple GPUs co-located on the same node (for NVLink bandwidth), use placement groups:

```python
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# All 4 GPUs must be on the same node (for NVLink)
pg = placement_group([{"GPU": 4}], strategy="STRICT_PACK")

@serve.deployment(
    ray_actor_options={
        "num_gpus": 4,
        "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=pg),
    }
)
class LargeModelDeploy:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM(
            model="meta-llama/Llama-3-70B-Instruct",
            tensor_parallel_size=4,   # 4-way tensor parallel across NVLink GPUs
        )
```

`STRICT_PACK` forces all resources onto a single node. `SPREAD` distributes them across nodes (useful for pipeline parallelism where cross-node bandwidth matters less than compute isolation). For 70B models with 4-way tensor parallelism, you need NVLink bandwidth (~600 GB/s on NVLink 4.0 vs ~25 GB/s on 100GbE), so `STRICT_PACK` is mandatory.

### Custom accelerator resources and heterogeneous clusters

Ray Serve works with any custom resource type that Ray can track. For specialized accelerators (Habana Gaudi, AWS Trainium/Inferentia, Google TPU), you register the resource and request it in `ray_actor_options`:

```python
# Register a custom resource type when starting the Ray worker node
# On the worker node:
# ray start --address=<head>:6379 --resources='{"Intel_Gaudi_HPU": 8}'

@serve.deployment(
    ray_actor_options={"resources": {"Intel_Gaudi_HPU": 1}},
)
class GaudiEmbedder:
    def __init__(self):
        # Habana Gaudi uses the habana_frameworks.torch module
        import habana_frameworks.torch as ht
        import torch
        self.model = torch.load("embedding_model.pt")
        ht.core.mark_step()  # move model to Gaudi HPU

    async def __call__(self, request):
        data = await request.json()
        # Gaudi inference
        import habana_frameworks.torch as ht
        result = self.model(data["input"])
        ht.core.mark_step()
        return {"output": result.tolist()}
```

This works identically for Intel CPUs with AMX extensions, AWS Inferentia2, and any FPGA-backed accelerator that exposes a Python interface. Ray Serve's resource model is intentionally hardware-agnostic — the scheduler matches `resources` dictionaries, not hardware types.

## 7. Request batching with `@serve.batch`

GPU inference is most efficient when you submit a batch of inputs in a single forward pass. A single `model(batch)` call for 16 inputs is far cheaper than 16 separate `model(x)` calls — kernel launch overhead, memory transfer setup, and CUDA stream synchronization all amortize over the batch.

The problem: HTTP requests arrive one at a time. Without batching, your GPU utilization can stay at 18–22% even under moderate load, because most of the time the GPU is waiting for the next request to be dispatched.

`@serve.batch` solves this by holding incoming requests in a short buffer and flushing them together as a single method call:

```python
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=128,
)
class BatchedEmbedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-mpnet-base-v2").cuda()

    @serve.batch(
        max_batch_size=64,
        batch_wait_timeout_s=0.05,   # wait up to 50ms to fill a batch
    )
    async def batch_encode(self, texts: list[str]) -> list[list[float]]:
        # Serve passes a LIST of texts from concurrent callers
        import torch
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                batch_size=len(texts),
                show_progress_bar=False,
            )
        # Return a list of results — Serve routes each result back to its caller
        return embeddings.cpu().tolist()

    async def __call__(self, request):
        data = await request.json()
        # When 10 concurrent requests each call batch_encode("some text"),
        # Serve collects all 10, calls batch_encode(["text1", ..., "text10"]),
        # and routes the results back to the correct awaiting callers.
        result = await self.batch_encode(data["text"])
        return {"embedding": result}
```

The mechanics: `@serve.batch` is an async coroutine decorator that intercepts individual calls to `batch_encode`. When multiple concurrent requests each call `self.batch_encode(text)`, Serve holds them in a FIFO buffer. After `batch_wait_timeout_s` elapses or `max_batch_size` items accumulate, Serve calls the underlying function once with a list of all buffered inputs. Each caller's `await` unblocks with its slice of the batch output. The return type must be a list of the same length as the input list — Serve zips inputs to outputs by position.

![Request batching before-after: serial serving with 16 kernel launches, 18% GPU utilization, 40 req/s vs @serve.batch with 1 kernel launch, 82% GPU utilization, 280 req/s on A100](/imgs/blogs/ray-serve-deep-dive-7.webp)

### Batching throughput math

Let $B$ be the batch size, $T_k$ be the kernel launch overhead (typically 0.5–2 ms on modern GPUs), and $T_c(B)$ be compute time for a batch of size $B$.

For transformer inference, $T_c(B)$ is memory-bandwidth-bound at small B and compute-bound at large B. A useful approximation for BERT-sized models:

$$T_c(B) \approx T_c(1) \times B^{0.7}$$

The throughput improvement from batching:

$$\text{Speedup}(B) = \frac{B}{T_k + T_c(B)} \div \frac{1}{T_k + T_c(1)} = \frac{B \cdot (T_k + T_c(1))}{T_k + T_c(B)}$$

For $T_k = 1.5$ ms and $T_c(1) = 3$ ms (small embedding model on A100): at B=64, $T_c(64) \approx 3 \times 64^{0.7} \approx 55$ ms. Speedup = $64 \times 4.5 / 56.5 \approx 5.1$×. For larger models where $T_c(1) = 25$ ms (BERT-large), at B=32: $T_c(32) \approx 25 \times 32^{0.7} \approx 235$ ms. Speedup = $32 \times 26.5 / 236.5 \approx 3.6$×.

The penalty is latency: the last request in a batch waits up to `batch_wait_timeout_s` plus the time to fill the batch. For a busy system filling batches in < 10 ms, the effective latency overhead is just the wait time. For a lightly loaded system where batches rarely fill, every request waits the full `batch_wait_timeout_s`.

The tuning rule: set `batch_wait_timeout_s` to at most **10% of your p99 SLA budget**. For a 500 ms SLA, 50 ms timeout is acceptable. For a 50 ms SLA, 5 ms timeout is the limit.

### Combining `@serve.batch` with PyTorch batch inference

For models where the forward pass itself supports variable-length batches (via attention masks or padding), the batch method can handle heterogeneous input lengths:

```python
@serve.batch(max_batch_size=32, batch_wait_timeout_s=0.02)
async def batch_classify(self, texts: list[str]) -> list[dict]:
    import torch
    # Tokenize as a batch, padded to max length in batch
    inputs = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        outputs = self.model(**inputs)
    logits = outputs.logits.cpu()
    # Return one result dict per input text
    return [
        {"label": self.id2label[l.item()], "score": float(s.max())}
        for l, s in zip(logits.argmax(-1), logits.softmax(-1))
    ]
```

The critical constraint: the return type must be a `list` with exactly `len(texts)` elements. If your model returns a tensor with shape `[B, ...]`, convert it to a Python list of per-sample results before returning. Serve cannot split a tensor — it needs a Python iterable to route results back to callers.

### Combining `@serve.batch` with dynamic batch size selection

A common production pattern is to allow clients to opt into batching by sending multiple inputs in one request, while also batching across concurrent single-input requests:

```python
@serve.deployment(ray_actor_options={"num_gpus": 1})
class FlexibleBatchClassifier:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.03)
    async def _batch_classify(self, texts: list[str]) -> list[dict]:
        import torch
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        scores = logits.softmax(-1).cpu()
        return [
            {"label": self.id2label[l.item()], "score": float(s.max())}
            for l, s in zip(logits.argmax(-1), scores)
        ]

    async def __call__(self, request):
        data = await request.json()
        texts = data["texts"] if isinstance(data["texts"], list) else [data["texts"]]
        # For multi-text requests, gather results and unstack
        results = await asyncio.gather(*[self._batch_classify(t) for t in texts])
        return {"results": results, "count": len(results)}
```

When a single client sends `{"texts": ["text1", "text2", "text3"]}`, this fans out three concurrent `_batch_classify` calls that all land in the same batch window alongside calls from other concurrent clients. The GPU sees a combined batch — maximally efficient regardless of whether clients send single texts or multiple.

### Backpressure and request queuing

`max_concurrent_queries` controls how many requests a single replica can hold in its queue at once. When this limit is reached, the HTTP proxy returns HTTP 503 (Service Unavailable) to new requests. This is backpressure — it prevents unbounded queue growth that would cause memory exhaustion and runaway latency.

Setting `max_concurrent_queries` requires understanding your deployment's characteristics:
- For compute-bound deployments (GPU forward pass): set to 2–4× `target_num_ongoing_requests_per_replica`. The autoscaler will add replicas before this limit is hit under normal operation; the limit protects against traffic spikes faster than autoscaling can respond.
- For I/O-bound deployments (database calls, external API): can be set much higher (50–200) because requests spend most of their time waiting, not consuming CPU.
- For streaming endpoints: set higher than the expected number of concurrent streaming sessions, since a streaming response occupies the replica's slot for the entire stream duration.

```python
@serve.deployment(
    max_concurrent_queries=8,    # reject with 503 if > 8 concurrent in-flight
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 4,  # scale up at 50% of limit
        "max_replicas": 16,
    },
)
class GeneratorWithBackpressure: ...
```

## 8. HTTP integration: FastAPI, streaming, and middleware

Ray Serve deployments expose a Starlette ASGI interface. Your `__call__` method receives a `starlette.requests.Request` object, and you return anything Starlette can serialize.

### FastAPI integration with `@serve.ingress`

The cleanest integration is `@serve.ingress(app)` — you write a normal FastAPI app and Serve wraps it, giving you Pydantic validation, OpenAPI documentation, and FastAPI middleware:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

app = FastAPI(title="Embedding Service", version="1.0")

class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True
    model: str = "all-MiniLM-L6-v2"

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    tokens_processed: int

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class EmbeddingService:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.models = {
            "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
            "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
        }

    @app.post("/v1/embed", response_model=EmbedResponse)
    async def embed(self, req: EmbedRequest) -> EmbedResponse:
        model = self.models.get(req.model, self.models["all-MiniLM-L6-v2"])
        embeddings = model.encode(req.texts, normalize_embeddings=req.normalize).tolist()
        total_tokens = sum(len(t.split()) for t in req.texts)  # approximate
        return EmbedResponse(embeddings=embeddings, model=req.model, tokens_processed=total_tokens)

    @app.get("/health")
    async def health(self):
        return {"status": "ok", "replicas": 2}
```

### Streaming responses for LLMs

For LLM token streaming, return a `StreamingResponse` with SSE format:

```python
from starlette.responses import StreamingResponse

@serve.deployment(ray_actor_options={"num_gpus": 1})
class StreamingGenerator:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            device_map="cuda",
            torch_dtype="float16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    async def __call__(self, request):
        data = await request.json()
        prompt = data["prompt"]

        async def token_generator():
            from transformers import TextIteratorStreamer
            import threading
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            def run_generate():
                self.model.generate(**inputs, max_new_tokens=256, streamer=streamer)

            thread = threading.Thread(target=run_generate, daemon=True)
            thread.start()
            for token in streamer:
                yield f"data: {token}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable Nginx buffering
                "Connection": "keep-alive",
            },
        )
```

The `X-Accel-Buffering: no` header is critical when Nginx or any reverse proxy sits in front of Serve. Without it, Nginx buffers the entire SSE stream and the user sees a delayed wall of text instead of per-token streaming. This header tells Nginx (and compatible proxies like Envoy with the right configuration) to flush each chunk immediately.

### gRPC support

As of Ray 2.9, Serve has native gRPC support. You define a proto, implement the servicer in a Serve deployment, and expose both HTTP and gRPC from the same cluster:

```python
# myservice.proto
# service EmbeddingService {
#   rpc Encode (EncodeRequest) returns (EncodeResponse);
# }

from ray.serve.drivers import gRPCIngress
import myservice_pb2, myservice_pb2_grpc

@serve.deployment(num_replicas=2)
class GRPCEmbeddingService(
    gRPCIngress, myservice_pb2_grpc.EmbeddingServiceServicer
):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    async def Encode(self, request, context):
        embeddings = self.model.encode(list(request.texts)).tolist()
        return myservice_pb2.EncodeResponse(
            embeddings=[myservice_pb2.Embedding(values=e) for e in embeddings]
        )
```

gRPC is faster than HTTP/JSON for high-throughput inter-service calls (Protocol Buffer serialization is ~5× faster than JSON for float arrays) and supports bidirectional streaming for token-by-token LLM output without the SSE framing overhead.

### Middleware for authentication and tracing

Starlette middleware applies to every request entering the deployment:

```python
from starlette.middleware.base import BaseHTTPMiddleware
import time

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        response.headers["X-Request-Duration-Ms"] = str(round(duration_ms, 1))
        return response

app = FastAPI()
app.add_middleware(MetricsMiddleware)

# Apply middleware: wrap app with serve.ingress AFTER adding middleware
@serve.deployment
@serve.ingress(app)
class MyService: ...
```

## 9. Fault tolerance: health checks, restart policies, and circuit breakers

A GPU OOM crash in one replica should not take down the entire service. Ray Serve handles this through its replica health-check loop.

![Fault tolerance: health check detects replica crash, router sheds traffic to live replicas while Ray restarts the dead actor, new replica loads model weights, and rejoins the pool in normal serving](/imgs/blogs/ray-serve-deep-dive-8.webp)

Every deployment has two health-check parameters:
- `health_check_period_s` (default 10): Serve calls a `check_health()` method on each replica every N seconds.
- `health_check_timeout_s` (default 30): if a replica does not respond within this window, it is declared dead and scheduled for restart.

```python
@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,
        "max_restarts": 5,        # restart up to 5 times before giving up
        "max_task_retries": 3,    # retry individual method calls on exception
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    graceful_shutdown_timeout_s=20,  # finish in-flight requests before shutdown
)
class RobustGenerator:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM(model="meta-llama/Llama-3-8B-Instruct", gpu_memory_utilization=0.90)
        self._healthy = True

    def check_health(self):
        # Raise to signal unhealthy; Serve will restart the replica
        if not self._healthy:
            raise RuntimeError("Replica entered unhealthy state")
        # Optional: check GPU health
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=ecc.errors.uncorrected.volatile.total",
                                  "--format=csv,noheader"], capture_output=True, text=True)
        if result.returncode == 0 and int(result.stdout.strip() or "0") > 0:
            raise RuntimeError("Uncorrected ECC errors detected")

    async def __call__(self, request):
        data = await request.json()
        try:
            outputs = self.llm.generate([data["prompt"]])
            return {"output": outputs[0].outputs[0].text}
        except Exception as e:
            self._healthy = False  # signal check_health to restart
            raise
```

The worst-case recovery window is `health_check_period_s + health_check_timeout_s = 40 seconds`. During this window, Serve routes all traffic to the remaining live replicas. If you have 4 replicas and 1 dies, the surviving 3 must absorb the full load. Design your replica count to handle `1/num_replicas` loss at peak traffic:

$$N_{min} = \left\lceil N_{target} \times \frac{\text{peak QPS}}{\text{throughput per replica}} \times \frac{1}{1 - 1/N_{target}} \right\rceil$$

For `N_target = 4` and 5% overhead headroom, you need `4 × 1.33 ≈ 6` replicas to maintain SLA during one failure with full peak load. Plan for failure.

### Replica crash root causes and prevention

In production, replica crashes cluster around four root causes:

**1. CUDA OOM**: the most common failure mode for GPU replicas. A request arrives with an unusually long input — a 16,000-token document when your model is configured for 4,096 max tokens. The model tries to allocate KV cache memory and fails. Mitigation: validate input lengths at the HTTP layer before they reach the GPU, and set `max_model_len` explicitly in vLLM or equivalent.

**2. Python OOM**: large intermediate tensors, unbounded list accumulation in long-running replicas (reference leak), or a single massive batch that exceeds CPU RAM. Mitigation: set `memory` in `ray_actor_options` to give the replica a soft memory limit and trigger a restart before the node OOM-kills it.

**3. Zombie processes**: a replica gets stuck in an infinite loop (infinite generation, stuck tokenizer) and stops responding to health checks. The `health_check_timeout_s` catches this — after the timeout, Serve kills and restarts the replica. Mitigation: add explicit `asyncio.wait_for(timeout=...)` wrappers around model calls with a timeout < `health_check_timeout_s`.

**4. CUDA context corruption**: rare but catastrophic — a replica's CUDA context enters an invalid state after a previous OOM or hardware ECC error. Subsequent calls fail with cryptic CUDA errors. The `check_health()` method can detect this by running a lightweight test forward pass on a fixed input:

```python
def check_health(self):
    import torch
    try:
        # Run a tiny test forward pass to verify CUDA context is healthy
        test_input = torch.zeros(1, 16, dtype=torch.long).cuda()
        with torch.no_grad():
            _ = self.model.embed_tokens(test_input)
    except Exception as e:
        raise RuntimeError(f"CUDA context corrupted: {e}")
```

## 10. Observability: metrics, tracing, and the Ray dashboard

Ray Serve exports Prometheus metrics automatically to `localhost:8080/metrics`. The key metrics for operational monitoring:

| Metric | Type | What it tells you |
|---|---|---|
| `ray_serve_num_ongoing_requests` | Gauge | In-flight requests per deployment |
| `ray_serve_deployment_queued_queries` | Gauge | Requests waiting for a replica |
| `ray_serve_http_request_latency_ms` | Histogram | End-to-end HTTP latency |
| `ray_serve_deployment_autoscaling_decisions` | Counter | Scale up/down events |
| `ray_serve_replica_health_check_duration_ms` | Histogram | Health check round-trip |
| `ray_serve_replica_processing_queries` | Gauge | Active queries per replica |

```yaml
# prometheus/scrape-configs.yaml
scrape_configs:
  - job_name: "ray_serve"
    static_configs:
      - targets: ["ray-head:8080"]
    scrape_interval: 15s

# Grafana alerting rule
groups:
  - name: ray_serve
    rules:
      - alert: ServeQueueBacklog
        expr: |
          ray_serve_deployment_queued_queries{deployment="GeneratorDeploy"} > 20
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GeneratorDeploy queue > 20 for 2 min — increase max_replicas or reduce request rate"
```

### Structured logging with request IDs

```python
import logging, uuid
logger = logging.getLogger("ray.serve")

@serve.deployment
class TracedService:
    async def __call__(self, request):
        request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        logger.info("request_started", extra={"request_id": request_id, "path": str(request.url)})
        try:
            result = await self._process(request)
            logger.info("request_ok", extra={"request_id": request_id})
            return result
        except Exception as e:
            logger.error("request_error", extra={"request_id": request_id, "error": str(e)})
            raise
```

Ray logs go to the head node's `~/.ray/logs/` directory by default. In Kubernetes, use a sidecar log aggregator (Fluentd, Vector) to ship structured JSON logs to Elasticsearch or CloudWatch. The `ray_serve` logger uses Python's standard `logging` module, so any handler you configure (JSON formatter, remote sink) works transparently.

### Grafana dashboards for Serve

A production-ready Grafana dashboard for Ray Serve should show:

1. **Per-deployment queue depth** (`ray_serve_deployment_queued_queries`): the primary autoscaling signal. If this is consistently > 0, you are under-replicated.
2. **Request latency histogram** (`histogram_quantile(0.99, ray_serve_http_request_latency_ms_bucket)`): your SLO metric. Alert when p99 exceeds your SLA budget.
3. **Replica count over time** (`ray_serve_num_deployment_replicas_available`): visualize autoscaling behavior — see the scale-up events and their latency.
4. **Error rate** (`rate(ray_serve_http_requests_total{status_code!~"2.."}[5m])`): HTTP errors, timeouts, and 503 backpressure events.

```promql
# Autoscaling effectiveness: ratio of queued to serving requests
ray_serve_deployment_queued_queries{deployment="GeneratorDeploy"}
  /
(ray_serve_num_deployment_replicas_available{deployment="GeneratorDeploy"} *
 on() group_left() scalar(
   sum(ray_serve_replica_processing_queries{deployment="GeneratorDeploy"})
   /
   count(ray_serve_replica_processing_queries{deployment="GeneratorDeploy"})
 ))
```

A ratio > 1.0 means the queue is growing faster than it is draining — autoscaling has not caught up yet, or you have hit `max_replicas`. Time this alert to fire after 60 seconds of ratio > 1.0 to avoid noise from transient bursts.

## 11. Ray Serve vs the alternatives: decision matrix

![Ray Serve vs alternatives: matrix comparing Ray Serve, Triton, vLLM, TorchServe, BentoML across Python flexibility, raw GPU throughput, autoscale-to-zero, multi-model pipeline, and LLM continuous batching](/imgs/blogs/ray-serve-deep-dive-5.webp)

The matrix is the honest picture. No tool dominates all dimensions:

**Ray Serve wins** when:
- Your serving logic is non-trivial Python — FAISS retrieval, NumPy preprocessing, HuggingFace tokenizers, OpenCV
- You need independent autoscaling per stage in a multi-model pipeline
- Your team already runs Ray for distributed training
- Latency SLA is > 200 ms (Python actor overhead is < 0.5% of the budget)
- Scale-to-zero for cost savings is a priority

**Triton wins** when:
- You need maximum raw GPU throughput for a single model (C++ execution engine, CUDA graph replay, ~1.5–2× higher throughput than Python actor models)
- Your model has a Triton backend (TensorRT, ONNX Runtime, PyTorch, TensorFlow)
- Your preprocessing is simple enough to express in Triton's C++ preprocessing pipelines

**vLLM wins** when:
- Your workload is LLM generation and you need continuous batching, PagedAttention, and prefix caching
- You have a strict latency SLA for LLM output (vLLM's continuous batching handles SLA variability better than static batching)
- You want LLM-specific features: speculative decoding, LoRA adapter hot-swap, tensor parallelism for 70B+ models

**TorchServe wins** when:
- Your organization is Java/Kotlin-based and needs JVM-compatible tooling
- You need the TorchServe management API and its Kubernetes operator
- Your team has existing TorchServe expertise

**BentoML wins** when:
- You want the simplest possible path from Python class to Docker image to cloud deployment
- You do not need cluster-level autoscaling or multi-node distribution
- Your team has no Ray or Kubernetes experience and the deadline is tight

The practical pattern at scale: **vLLM inside Ray Serve**. vLLM handles LLM inference internals; Ray Serve handles routing, pipeline composition, and autoscaling. The two tools compose naturally:

```python
@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 4},
    ray_actor_options={"num_gpus": 1},
)
class VLLMDeploy:
    def __init__(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        args = AsyncEngineArgs(
            model="meta-llama/Llama-3-8B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def __call__(self, request):
        from vllm import SamplingParams
        data = await request.json()
        sampling = SamplingParams(temperature=0.7, max_tokens=data.get("max_tokens", 256))
        results = self.engine.generate(data["prompt"], sampling, str(id(request)))
        async for output in results:
            if output.finished:
                return {"output": output.outputs[0].text}
```

## 12. When to use Ray Serve (and when not to)

This section is blunt. Choosing the wrong tool adds months of pain.

**Use Ray Serve when:**

Your serving logic is Python and it is not trivially wrappable in C++. Five Python functions chained together, where each calls a different library, is exactly the use case Ray Serve was built for. You need per-stage autoscaling — the embedder and the generator scale at fundamentally different rates. You are already on Ray for training and want to reuse the cluster and infrastructure. Your p99 SLA is > 200 ms. You want scale-to-zero without building a custom Kubernetes operator.

**Do NOT use Ray Serve when:**

You need maximum throughput from a single model and the preprocessing is trivial — use Triton. You are serving a large LLM and need continuous batching, PagedAttention, and speculative decoding as first-class features — use vLLM directly, or wrap vLLM in Serve for pipeline composition. Your team has no Ray experience and your deadline is two weeks — BentoML or a plain Kubernetes + Triton deployment will be faster to production. Your p99 SLA is < 50 ms — Python actor overhead and the DeploymentHandle round-trips (~200–400 µs each) are measurable fractions of the budget. Your organization runs JVM-based services and expects Java-compatible tooling.

#### Worked example: choosing between Triton and Ray Serve for a vision API

You run a computer vision API classifying uploaded images with ResNet-50. Traffic: 500 req/s peak. SLA: p99 < 50 ms. No preprocessing beyond resize and normalize.

- **Triton**: drop-in model repository, dynamic batching, CUDA graph replay. Single-model throughput on T4 16GB: ~800 req/s at batch 32, p99 ~30 ms. Passes the SLA easily.
- **Ray Serve**: `@serve.deployment` wrapping PyTorch, `@serve.batch` for batching. Single-model throughput on T4 16GB: ~550 req/s at batch 32, p99 ~40 ms. Python actor overhead costs ~30% throughput.

Now add a requirement: after classification, run an optional OCR model on detected text regions, routing results to different downstream pipelines depending on the classification label. Now you have two models, conditional routing, and different scaling curves.

- **Triton**: Triton ensembles can express conditional routing in C++ backends, but OCR typically uses Tesseract or EasyOCR — Python libraries without Triton backends. You would need a Python preprocessing backend, reintroducing Python overhead.
- **Ray Serve**: natural fit. ClassifierDeploy and OCRDeploy are separate deployments. The router makes the conditional call via DeploymentHandle. Each scales independently. The 400 µs handle overhead is invisible against the 50 ms classification + 80 ms OCR.

As pipeline complexity grows, Ray Serve's Python-first model becomes an advantage.

### The Kubernetes + Ray Serve integration decision

One question that comes up frequently: should you deploy Ray Serve on Kubernetes, or should you use Kubernetes-native ML serving (KServe, Seldon, BentoML Cloud)?

Ray Serve on Kubernetes (via KubeRay) gives you the best of both worlds: Ray's Python-native multi-deployment composition and autoscaling, plus Kubernetes' infrastructure maturity (node autoscaling, persistent volumes, network policies, RBAC, Helm). The cost is operational complexity — you run a Ray cluster **inside** Kubernetes, which means two layers of orchestration to understand.

The alternative — KServe or Seldon — uses Kubernetes-native custom resources (`InferenceService`, `SeldonDeployment`) to express serving graphs. These have tighter Kubernetes integration (Istio service mesh, Knative for scale-to-zero, Prometheus ServiceMonitor CRDs) but more limited expressiveness for Python-heavy pipelines. If your serving logic is "model in, prediction out" with no custom Python, KServe is simpler. If your serving logic involves custom Python at every stage, Ray Serve on KubeRay is more expressive.

The production decision matrix:

| Requirement | Recommendation |
|---|---|
| Python pipelines with 3+ stages | Ray Serve on KubeRay |
| Single model, no custom Python | KServe or Triton |
| Existing Istio mesh investment | KServe (native Istio integration) |
| Team already on Ray for training | Ray Serve (reuse cluster) |
| Knative scale-to-zero | KServe (native Knative) |
| Budget-sensitive dev/staging | Ray local mode + minikube |

#### Worked example: per-request cost for a 7B model at different throughput levels

Suppose you are choosing between a dedicated GPU instance and a Ray Serve autoscaling deployment for a chatbot serving 7B-parameter Mistral 7B. The cost per request depends heavily on throughput utilization:

| Configuration | Idle cost/hr | Peak throughput | P50 cost/1k requests |
|---|---|---|---|
| Dedicated `g5.xlarge` always-on | \$1.006 | 5 req/s | \$0.056 |
| Ray Serve, `min_replicas=2`, `max=8` | \$2.012 (2 GPU) | 40 req/s | \$0.014 |
| Ray Serve, `min_replicas=0`, keep-warm ping | \$1.006 (1 GPU overnight) | 40 req/s burst | \$0.007 avg |

The autoscaling configuration is 8× cheaper per request at peak load, because it spreads the burst across 8 replicas instead of queuing behind 1. At low load, scale-to-zero with keep-warm ping nearly matches dedicated cost. The break-even point is roughly 60% GPU utilization — above that, autoscaling per-request costs are lower; below that, dedicated is comparable.

### Practical migration: from a FastAPI monolith to Ray Serve

The most common migration path onto Ray Serve starts with a FastAPI application that loads a single model at startup and handles requests serially. As traffic grows, the FastAPI app becomes a bottleneck: one Python process, one model, no batching, manual scaling via Kubernetes replicas with a fixed model-per-pod ratio.

The migration to Ray Serve is incremental. Start by wrapping the existing FastAPI app inside a `@serve.deployment` without changing any serving logic — this immediately adds autoscaling and concurrent replica management. The model is still loaded once per replica; requests still run one at a time per replica; but replicas now scale on queue depth instead of CPU utilization, and the Ray cluster manages placement across multiple GPU nodes.

Next, add `@serve.batch` to the inference endpoint. This requires changing the function signature from `(self, request)` to `(self, requests: list)` and batching the preprocessing step — typically a 20-line change. The throughput improvement is immediate: the GPU shifts from memory-bandwidth-bound (batch 1) to compute-bound (batch 16+).

Finally, split the preprocessing into a separate `@serve.deployment` if it is CPU-heavy (e.g. PDF parsing, audio transcription). Now the CPU preprocessing stage and the GPU inference stage scale independently, and the cluster can spin up 4 preprocessing replicas per 1 GPU replica during document-heavy traffic spikes.

This three-step migration — wrap, batch, split — covers 80% of FastAPI → Ray Serve migrations and can be executed without downtime using Ray Serve's hot-reload capability (`serve.run()` updates deployments in-place).

## 13. Benchmarks and case studies

### Case study A: multi-model RAG serving at a production AI startup

A company serving 20 million RAG queries per day migrated from six separate FastAPI microservices in Kubernetes to Ray Serve. Pipeline: query parsing → embedding → FAISS retrieval → cross-encoder reranking → generation (Mistral 7B).

Before: 6 separate Docker images, 6 Kubernetes Deployments, 6 HPAs, 6 independent log streams. Inter-service HTTP calls via ClusterIP added 15–30 ms per hop. Pipeline p99 latency: 300 ms. Infrastructure team spent ~40% of time on deployment incidents.

After: one Ray cluster, one Serve application, five deployments. Handle-call overhead replaces HTTP calls: ~0.4 ms per hop vs 15–30 ms. Pipeline p99 improved to 220 ms — 80 ms reduction from eliminating inter-service HTTP alone. Infrastructure overhead dropped by 60%.

### Case study B: embedding API batching at scale

An embedding API provider serving ~1,536-dimensional embeddings (BERT-large architecture) enabled `@serve.batch` with `max_batch_size=32, batch_wait_timeout_s=0.02`.

| Metric | Before batching | After batching |
|---|---|---|
| GPU utilization | 22% | 79% |
| Throughput (req/s per A100) | 85 | 420 |
| p50 latency | 18 ms | 34 ms (+16 ms wait) |
| p99 latency | 45 ms | 62 ms (+17 ms wait) |
| Cost per 1M embeddings | \$0.42 | \$0.085 |

The 20 ms batch wait roughly doubles p50 latency but drops cost per million embeddings by 5×. For background indexing workloads, this is an excellent trade.

### Case study C: scale-to-zero for LLM evaluation harnesses

An NLP research group runs nightly evals against 30 fine-tuned Llama-3-8B checkpoints. Historically, two GPU nodes ran warm 24/7 for eval queuing — \$29/day.

With `min_replicas=0`, each endpoint scales down after 10 minutes of inactivity. The nightly eval harness cold-starts each endpoint on demand, processes its eval suite in ~3 minutes, then the endpoint scales to zero. Total active GPU-hours per night: ~1.5 GPU-hours across all 30 checkpoints. Cost: \$1.51/night vs \$29/night — a 19× reduction.

### Case study D: Kubernetes deployment of a Ray Serve cluster

For Kubernetes deployments, the KubeRay operator is the standard approach. It manages the lifecycle of the Ray cluster as Kubernetes custom resources:

```yaml
# raycluster.yaml — KubeRay cluster spec
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: rag-serving-cluster
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-cpus: "4"
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray-ml:2.9.0-gpu
            resources:
              requests:
                cpu: "4"
                memory: "8Gi"
            ports:
              - containerPort: 6379   # GCS
              - containerPort: 8265   # Dashboard
              - containerPort: 8000   # Serve HTTP
  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 4
      rayStartParams:
        num-gpus: "1"
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray-ml:2.9.0-gpu
              resources:
                requests:
                  cpu: "8"
                  memory: "32Gi"
                  nvidia.com/gpu: "1"
                limits:
                  nvidia.com/gpu: "1"
```

```bash
# Deploy the cluster
kubectl apply -f raycluster.yaml

# Wait for cluster to be ready
kubectl wait --for=condition=Ready raycluster/rag-serving-cluster --timeout=300s

# Port-forward to access the Serve endpoint locally
kubectl port-forward service/rag-serving-cluster-head-svc 8000:8000 &

# Deploy your Serve application to the running cluster
RAY_ADDRESS="ray://localhost:10001" python -c "
from ray import serve
import ray
ray.init(address='auto')
serve.start(detached=True)
# Then serve.run(your_app, ...)
"
```

The KubeRay operator handles GPU node scaling via Cluster Autoscaler integration — when Ray Serve's autoscaler adds replicas and the cluster lacks GPU nodes, Cluster Autoscaler provisions new EC2/GKE/AKS nodes to accommodate them. This end-to-end scaling works without any manual intervention.

### Benchmark: DeploymentHandle overhead by hop count

| Pipeline stages | Total handle overhead | Recommendation |
|---|---|---|
| 1 stage (direct model) | ~0 ms | Any framework |
| 3 stages (embed → retrieve → generate) | ~0.8 ms | Negligible for 200 ms pipelines |
| 7 stages (complex RAG) | ~2.4 ms | Acceptable for > 50 ms pipelines |
| 15 stages (microservices-heavy) | ~5 ms | Consider stage merging |

For very deep pipelines (> 10 stages), merge compute-cheap stages into single deployments to reduce hop count. A "preprocessor" deployment that runs embedding + retrieval + reranking as sequential Python code adds zero handle overhead between those steps.

### Case study E: cold start optimization with pre-warming

A common trick to reduce cold start time for GPU models is to pre-download model weights to the node's local NVMe before the replica launches. This converts a cold start from "download 14 GB from S3 over network" (3–5 minutes) to "read 14 GB from local SSD" (15–45 seconds).

With KubeRay and Kubernetes DaemonSets, you can run a weight-download DaemonSet that populates a local cache path on every GPU node:

```yaml
# weight-downloader-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: model-weight-downloader
spec:
  selector:
    matchLabels:
      app: weight-downloader
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/accelerator: "nvidia-a100"
      initContainers:
        - name: download-weights
          image: amazon/aws-cli:latest
          command:
            - sh
            - -c
            - |
              aws s3 sync s3://my-models/mistral-7b-v0.2/ /model-cache/mistral-7b-v0.2/
          volumeMounts:
            - name: model-cache
              mountPath: /model-cache
      containers:
        - name: placeholder
          image: busybox
          command: ["sleep", "infinity"]
      volumes:
        - name: model-cache
          hostPath:
            path: /opt/model-cache
            type: DirectoryOrCreate
```

Then in your Serve deployment:

```python
@serve.deployment(ray_actor_options={"num_gpus": 1})
class FastStartGenerator:
    def __init__(self):
        import os
        from vllm import LLM
        # Point vLLM to the pre-downloaded local cache
        local_path = "/opt/model-cache/mistral-7b-v0.2"
        if os.path.exists(local_path):
            model_path = local_path  # fast: local NVMe
        else:
            model_path = "mistralai/Mistral-7B-Instruct-v0.2"  # slow: HuggingFace download
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.90)
```

This reduces cold start from 3–5 minutes (network download) to 15–45 seconds (local NVMe read) — a 4–10× improvement that makes scale-to-zero viable for more workloads.

### Case study F: multi-modal serving with vision + text

A research lab serving a visual question answering (VQA) system — given an image and a question, produce an answer — built a three-deployment pipeline:

```python
@serve.deployment(num_replicas=4, ray_actor_options={"num_cpus": 2})
class ImageProcessor:
    def __init__(self):
        from PIL import Image
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    async def __call__(self, request):
        import io, base64, torch
        data = await request.json()
        img_bytes = base64.b64decode(data["image_b64"])
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        return tensor.numpy().tolist()  # serialize as JSON

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class VisionEncoder:
    def __init__(self, image_processor: DeploymentHandle):
        import torch
        from transformers import CLIPModel
        self._processor = image_processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

    async def encode_image(self, request):
        data = await request.json()
        pixel_values_ref = await self._processor.remote(request)
        pixel_values = await pixel_values_ref
        import torch
        pv = torch.tensor(pixel_values).cuda()
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pv)
        return features.cpu().tolist()

@serve.deployment(
    num_replicas=2,
    autoscaling_config={"min_replicas": 1, "max_replicas": 6},
    ray_actor_options={"num_gpus": 1},
)
class VQAAnswerer:
    def __init__(self, vision_encoder: DeploymentHandle):
        from transformers import pipeline
        self._encoder = vision_encoder
        self.qa_pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen-VL-Chat",
            device=0,
            torch_dtype="float16",
        )

    async def __call__(self, request):
        data = await request.json()
        # Get image features
        img_features_ref = await self._encoder.encode_image.remote(request)
        img_features = await img_features_ref
        # Combine with question and answer
        prompt = f"Image context: [visual features embedded]\nQuestion: {data['question']}\nAnswer:"
        answer = self.qa_pipe(prompt, max_new_tokens=128)[0]["generated_text"]
        return {"answer": answer, "image_processed": True}

# Build and deploy the VQA pipeline
processor = ImageProcessor.bind()
encoder = VisionEncoder.bind(image_processor=processor)
answerer = VQAAnswerer.bind(vision_encoder=encoder)
serve.run(answerer, route_prefix="/vqa")
```

The VQA system runs the CPU-bound image preprocessing on 4 cheap replicas, CLIP encoding on 2 GPU replicas, and the generation model on 2 autoscaling GPU replicas. Total infrastructure cost at idle: 4 GPU-hours/day. At peak: up to 8 GPU replicas, 8× higher throughput. No single serving framework outside Ray Serve could express this topology cleanly in pure Python.

## 14. Key takeaways

1. **The `@serve.deployment` decorator registers a recipe; `serve.run()` materializes it.** Circular dependencies fail at bind time, not runtime. This is intentional and correct.

2. **`serve.bind()` creates a directed acyclic computation graph.** Replicas launch in topological dependency order. The graph is your application's serving contract.

3. **Autoscaling is a queue-depth controller with hysteresis.** Tune `upscale_delay_s` to avoid thrashing on traffic spikes; tune `downscale_delay_s` to prevent cold starts during predictable daily traffic patterns (morning ramp, lunch dip, evening traffic).

4. **`@serve.batch` is the single highest-leverage optimization for memory-bandwidth-bound inference.** A 50 ms wait budget on an embedding model can deliver 15–20× throughput improvement. Tune `batch_wait_timeout_s` to < 10% of your p99 SLA budget.

5. **Fractional GPU allocation (`num_gpus: 0.5`) does not enforce VRAM isolation.** Use NVIDIA MIG on A100/H100 for real hardware isolation. Time-slicing works for trusted multi-tenant deployments where VRAM overflow is the caller's problem.

6. **Worst-case health-check recovery is 40 seconds** (`health_check_period_s + health_check_timeout_s`). Size replica pools to absorb one-in-N replica failures at peak load.

7. **DeploymentHandle call overhead is 200–400 µs per hop.** Negligible for LLM-heavy pipelines; measurable for pipelines with many sub-10 ms stages. Merge cheap stages to reduce hop count.

8. **Ray Serve and vLLM compose naturally.** Use vLLM's `AsyncLLMEngine` for LLM inference; wrap it in a `@serve.deployment` for routing, autoscaling, and pipeline composition.

9. **Scale-to-zero with a keep-warm ping is the dominant cost strategy for non-interactive endpoints.** Eliminate overnight idle GPU cost while maintaining zero cold-start risk during business hours.

10. **The `check_health()` method is your circuit-breaker hook.** Use it to detect stuck model states that Python exception handling would not catch — ECC errors, CUDA context corruption, exhausted VRAM fragmentation.

11. **KubeRay is the production path for Kubernetes deployments.** It manages Ray cluster lifecycle as Kubernetes CRDs, integrates with Cluster Autoscaler for node provisioning, and lets you use GitOps workflows for serving config. The complexity cost is two layers of orchestration to understand and debug.

12. **Pre-warming node caches reduces cold start 4–10×.** A DaemonSet that downloads model weights to local NVMe before replicas need them converts "download 14 GB over network" into "read 14 GB from disk". Essential for scale-to-zero in latency-sensitive environments.

13. **`max_concurrent_queries` is your backpressure valve.** Set it high enough to absorb bursts faster than the autoscaler can respond, but low enough to prevent runaway memory usage during traffic spikes. A good starting point: `3–5 × target_num_ongoing_requests_per_replica`.

14. **The Kubernetes + Ray Serve combination covers the full cost-performance spectrum.** Dev environments run Ray locally (no cluster overhead). Staging runs on a small KubeRay cluster with scale-to-zero. Production runs on autoscaling KubeRay with Cluster Autoscaler integration. The same application code and Serve configuration works at all three layers with only the cluster address changing.

## Further reading

- [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — the SLO triangle (latency, throughput, cost) that motivates every design decision in this post
- [Triton Inference Server deep dive](/blog/machine-learning/model-serving/triton-inference-server-deep-dive) — when maximum GPU throughput from a single model is the priority; Triton's ensemble pipelines as the alternative to Serve for C++-friendly workloads
- [Batching fundamentals: latency-throughput tradeoffs](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) — Little's Law and the queuing theory behind why batching works; how to derive the optimal `max_batch_size` for your SLA
- [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) — PagedAttention, continuous batching, prefix caching, and speculative decoding; the right engine to wrap inside a Ray Serve deployment for LLM workloads
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — the series capstone: a full decision tree from model type to production serving stack, with Ray Serve's position in the landscape
- Ray Serve official documentation at `docs.ray.io/en/latest/serve/` — comprehensive reference for all autoscaling config parameters, deployment options, and advanced features including gRPC support and multi-application namespacing
- "Ray: A Distributed Framework for Emerging AI Applications" (Moritz et al., OSDI 2018) — the foundational paper describing the actor model and object store that Ray Serve builds on
- Anyscale Engineering Blog — production case studies for Ray Serve deployments, vLLM integration patterns, and scale-to-zero cost optimization
