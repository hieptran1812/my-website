---
title: "Ray Serve in Production: Architecture, Best Practices, and a Practical Operating Guide"
date: "2026-10-01"
publishDate: "2026-10-01"
description: "Learn how Ray Serve routes requests, composes models, controls overload, scales replicas, and supports reliable inference operations."
tags: ["ray-serve", "model-serving", "inference", "mlops", "autoscaling", "distributed-systems", "rag", "observability", "kubernetes", "python"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 50
---

A model can be fast in a notebook and still make a terrible service. The model is only one station in a larger system: requests must be admitted, routed, queued, processed, timed out, measured, and sometimes rejected. A retrieval-augmented generation service makes this especially visible. The embedding stage wants cheap parallel CPU capacity, retrieval may wait on an external index, reranking can profit from batches, and generation consumes scarce GPU memory. Putting all four behind one process forces them to share one scaling decision.

Ray Serve gives us a way to express those stages as Python deployments and scale them independently on a Ray cluster. It also gives us enough knobs to hurt ourselves. A large concurrency limit can hide a saturated model behind a growing queue. More replicas cannot help when the cluster cannot place them. An `async def` handler does not make synchronous inference nonblocking. The useful skill is not memorizing decorators. It is understanding which layer owns each bottleneck.

![A request travels through a Serve proxy and router to a replica while the controller manages deployment state](/imgs/blogs/ray-serve-production-engineering-1.webp)

The diagram above is the mental model for this guide. Follow the solid request path when investigating latency. Follow the control path when investigating placement, recovery, or configuration. This article works from that distinction toward a small runnable service, then toward production decisions. It is based on the [Ray Serve 2.58.0 documentation](https://docs.ray.io/en/latest/serve/index.html), checked on October 1, 2026. Names and defaults should be checked again when upgrading Ray.

| Familiar assumption | What actually matters in production | First thing to inspect |
| --- | --- | --- |
| A faster model fixes a slow endpoint | Queueing, network calls, serialization, and cold starts also contribute | Per-stage latency and queue length |
| Autoscaling absorbs every spike | New replicas and possibly new nodes take time to start | Replica demand, placement, and startup time |
| `async def` means parallel inference | It only helps when the handler yields rather than blocking its event loop | Blocking calls and CPU/GPU saturation |
| More batching always saves money | Batch formation can consume the p99 latency budget | Throughput and latency at the same offered load |
| A retry is harmless | Retries can amplify an overloaded service | Retry rate, timeouts, and admission policy |

If you want a broad introduction to serving layers first, read [the model serving stack](/blog/machine-learning/inference-frameworks/the-model-serving-stack). An earlier [Ray Serve deep dive](/blog/machine-learning/inference-frameworks/ray-serve-deep-dive) covers the API in more breadth. Here we will keep asking a narrower operational question: what should we measure and change when the service is under pressure?

## 1. Decide what Ray Serve should own

**Rule of thumb: choose Serve when the application graph and its scaling behavior are the problem.** A single model on one machine may be well served by a simpler server. A service that coordinates several Python stages with different resource requirements is where Serve becomes compelling.

Ray Serve is a framework-agnostic online serving layer on top of Ray. A deployment wraps a Python class or function. Each replica is a Ray actor running the deployment code. An application composes deployments, usually by binding one deployment into another and calling the resulting `DeploymentHandle`. Ray schedules replicas on cluster resources; Serve exposes them over HTTP or gRPC and manages request routing and deployment state. The [official overview](https://docs.ray.io/en/latest/serve/index.html) explicitly emphasizes model composition, multi-model serving, and flexible scheduling.

Do not confuse this with a model execution engine. Serve can host an ordinary Python predictor, call a vector database, wrap a PyTorch model, or integrate with an engine such as vLLM. It does not make a model kernel faster by itself. If the main limitation is tokens per second inside one LLM engine, optimize the engine and its batching, memory, and parallelism first. If the limitation is coordinating several services, scaling them separately, or operating a multi-node endpoint, Serve is in its element. Ray's own [comparison section](https://docs.ray.io/en/latest/serve/index.html) states that Serve does not perform model-specific optimization on its own.

A useful kitchen analogy is to treat each deployment as a station with its own crew. A prep station can scale to ten workers without buying ten ovens. A GPU station can keep expensive ovens busy without making the entire kitchen GPU-backed. The proxy is the service counter, the router chooses a worker at a station, and the controller decides how many stations and workers should exist. The analogy breaks when we pretend the manager handles every plate. The manager is a control-plane component, not a mandatory hop in every request.

Before adopting Serve, write down four numbers for each stage: service time per request, achievable throughput per replica, memory or GPU footprint per replica, and startup time. Also write down whether the stage waits on I/O or occupies a CPU or GPU. These quantities predict whether independent deployments will help. If every stage has the same resource profile and all calls are local and cheap, splitting them into Ray actors may add serialization and network overhead without buying useful scaling freedom.

| Workload | First option to evaluate | Why |
| --- | --- | --- |
| One lightweight Python model, modest traffic | FastAPI or another simple process server | Fewer moving parts and low routing overhead |
| Several Python models with different resource needs | Ray Serve | Per-deployment scaling and composition |
| LLM generation bottleneck inside one engine | vLLM or another inference engine | Engine-level batching and memory management |
| LLM endpoint spanning engines, nodes, and routes | Ray Serve LLM with an engine | Serving orchestration around engine instances |
| Offline processing of a large dataset | Ray Data or a batch system | Online request machinery is a poor fit |

The second-order cost is operational. Ray gives flexible actor placement, but now the cluster, Serve controller, proxy, replicas, node resources, and client retries are part of your failure model. Adopt that complexity for a specific scaling or composition benefit, not because the decorator looks convenient.

## 2. Follow one request through the system

**Rule of thumb: debug the data path and the control path separately.** Confusing them leads to the wrong latency story.

A Serve instance has a controller, one or more proxies, and deployment replicas. The controller manages creation, updates, deletion, and recovery. A proxy accepts HTTP or gRPC traffic. A request is matched to an application and deployment, waits for a suitable replica if necessary, and is forwarded by a router associated with the caller. A `DeploymentHandle` call uses the same general selection mechanism for calls between deployments. The [architecture guide](https://docs.ray.io/en/latest/serve/architecture.html) describes the proxy, controller, and replicas and the lifetime of a request. It also notes that the default HTTP proxy placement is on the head node; other placements can be configured. Do not assume there is one proxy on every node unless you set that policy.

Imagine 100 requests arrive in a second. The proxy parses them and hands them to the relevant deployment's routing path. Some are assigned immediately. Others wait because every candidate replica has reached its ongoing-request limit. If the caller's queue is also full, additional requests are rejected. The controller does not need to answer a network round trip for each of those 100 requests. It keeps deployment and replica state current so the routers know what is available.

This gives us a latency budget:

$$L_{end\text{-}to\text{-}end} = L_{ingress} + L_{queue} + L_{dispatch} + L_{application} + L_{response}.$$

This is an **explanatory accounting identity**, not a Ray implementation formula. A four-stage application further decomposes $L_{application}$ into stage work and calls between stages. If the model takes 80 ms but the endpoint takes 900 ms, a model-only profiler will miss the problem. If $L_{queue}$ dominates, look at admission and capacity. If $L_{application}$ dominates, inspect the stage itself. If startup dominates only during bursts, investigate autoscaling and placement.

A deployment handle is important because it avoids turning every internal stage into a public HTTP endpoint. The caller can pass ordinary Python arguments to another deployment. The bound handle identifies the target deployment, while Serve routes each `.remote()` call to a replica. This is a composition mechanism, not a guarantee of zero serialization or zero network cost. Large objects, cross-node placement, and repeated tiny calls still matter. The [model composition guide](https://docs.ray.io/en/latest/serve/model_composition.html) documents binding and handles.

| Question | Inspect | Typical wrong fix |
| --- | --- | --- |
| Is the proxy receiving traffic? | HTTP request rate and status codes | Add model replicas before confirming ingress |
| Are calls waiting for replicas? | Queue and ongoing-request metrics | Increase timeout without reducing queue delay |
| Are replicas running slowly? | Processing latency and CPU/GPU utilization | Add nodes when per-replica code is blocking |
| Are desired replicas missing? | `serve status`, Ray resources, placement events | Raise `max_replicas` again |
| Did a new revision cause cold starts? | Replica startup and deployment events | Blame the router |

A practical consequence follows: annotate traces and logs with the application, deployment, replica, request ID, and stage. Without those dimensions, a graph that says “p99 increased” is a symptom, not a diagnosis.

## 3. Build a small RAG service

**Rule of thumb: split stages only when their capacity or failure behavior differs.** A deployment boundary is an operational choice, not a requirement to turn every Python function into an actor.

![A RAG service separates embedding retrieval reranking and generation into stages with different resources](/imgs/blogs/ray-serve-production-engineering-2.webp)

We will use a deliberately small retrieval example. Its vocabulary and answers are toy data, so the code runs without a model download or GPU. It demonstrates the application graph and handle calls. Replace each toy stage with a real implementation only after you can measure the stage in isolation. Save this as `rag_app.py`:

```python
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request

DOCUMENTS = [
    "Ray Serve routes requests to deployment replicas.",
    "A Ray actor is a stateful Python worker.",
    "Autoscaling adds replicas when a deployment is busy.",
    "A bounded queue protects latency during overload.",
]


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Retriever:
    def search(self, question: str) -> list[str]:
        words = set(question.lower().split())
        ranked = sorted(
            DOCUMENTS,
            key=lambda doc: len(words.intersection(doc.lower().split())),
            reverse=True,
        )
        return ranked[:2]


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Answerer:
    def __init__(self, retriever: DeploymentHandle):
        self.retriever = retriever

    async def __call__(self, request: Request) -> dict[str, object]:
        payload = await request.json()
        question = str(payload["question"])
        documents = await self.retriever.search.remote(question)
        return {"question": question, "evidence": documents}


app = Answerer.bind(Retriever.bind())
```

Run it locally with a recent Ray installation:

```bash
python -m pip install 'ray[serve]==2.58.0'
serve run rag_app:app
curl -sS -X POST http://127.0.0.1:8000/ \
  -H 'content-type: application/json' \
  -d '{"question":"How does Serve route requests?"}'
```

The output contains an `evidence` list. It is not a generated answer, and it is not a real RAG benchmark. That distinction matters: a code sample that quietly downloads a 7B model or assumes a vector index exists is not runnable for most readers. This minimal graph demonstrates `bind()`, the constructor-injected `DeploymentHandle`, and an awaited `.remote()` call. The [quickstart](https://docs.ray.io/en/latest/serve/index.html) and [composition guide](https://docs.ray.io/en/latest/serve/model_composition.html) use the same primitives.

In a real service, embedding may be a separate deployment, retrieval may call an external vector store, reranking may batch candidates, and generation may use Serve LLM or a hosted model. Do not immediately copy the four-stage diagram into four processes. First ask whether a boundary changes one of these decisions: scaling, hardware, deployment cadence, failure isolation, or ownership. If none changes, a local function call is easier to operate.

Consider a hypothetical steady state of 60 queries per second. If retrieval takes 20 ms of CPU time, one continuously busy core has a rough capacity of 50 queries per second before overhead. Two CPU replicas might cover the load. If generation requires 250 ms of GPU service time, one serial GPU lane has a rough capacity of four queries per second. Real generation engines overlap and batch requests, so that estimate is only a starting bound, not a measured capacity claim. The point is that one shared replica count cannot express both requirements. Benchmark the actual engine at the intended prompt and output lengths.

The second-order consequence is fan-out. A top-level request may call retrieval once, rerank twenty candidates, then stream hundreds of output tokens. A deployment boundary can make scaling clearer, but it can also create many RPCs. Batch candidate calls where the model supports it and pass compact data across boundaries. Avoid shipping a full corpus, giant prompt template, or repeated model weights with every request.

## 4. Write deployments that behave under load

**Rule of thumb: a replica's handler must match its actual work, not the word `async` in a style guide.**

![A Serve deployment defines the scale unit while replicas become Ray actors placed on cluster nodes](/imgs/blogs/ray-serve-production-engineering-3.webp)

A class decorated with `@serve.deployment` is a recipe. `__init__` runs when each replica starts, so load model weights there once per replica. A replica is not a shared-memory thread of one global model instance. Ten replicas generally mean ten model instances and their associated memory. This can be exactly what you need for throughput, and exactly what causes an out-of-memory event when someone raises `max_replicas` without checking GPU capacity.

The [Ray Serve asyncio guide](https://docs.ray.io/en/latest/serve/advanced-guides/asyncio-best-practices.html) recommends `async def` for I/O-bound calls that can yield, and `def` or explicit offload for CPU-bound work. `async def` does not preempt a long synchronous function call. This handler looks concurrent but can occupy its event loop while the blocking call runs:

```python
import time
from ray import serve


@serve.deployment
class BlockingExample:
    async def __call__(self) -> str:
        time.sleep(2)  # Blocks the event loop; do not copy into production.
        return "done"
```

For a real remote call, use an async client and `await`. For CPU work, use a synchronous handler, a thread or process strategy appropriate to the library, or more replicas. For GPU work, determine whether the inference engine supports concurrent requests and batching. If it does, expose that concurrency deliberately. If it does not, merely setting a high ongoing-request limit can move the queue inside the replica rather than increase useful compute.

Resources are declared with `ray_actor_options`. This is a scheduling reservation, not a performance guarantee. A replica that requests one GPU can be placed only where one GPU is available. A replica that requests fractional GPU resources may be colocated with others, but that does not partition VRAM into hard slices. Measure combined peak memory and interference before relying on fractional allocation. The [resource allocation guide](https://docs.ray.io/en/latest/serve/resource-allocation.html) explains the scheduling model.

```python
from ray import serve


@serve.deployment(
    ray_actor_options={"num_cpus": 2, "num_gpus": 1},
    max_ongoing_requests=8,
)
class GPUModel:
    def __init__(self):
        self.model = load_model_once()  # Supply your actual loader.

    def __call__(self, payload: dict) -> dict:
        return self.model.predict(payload)
```

The snippet above is a configuration pattern, not a standalone program: `load_model_once()` depends on your chosen model. The earlier toy RAG application is the runnable starting point. Be explicit about such boundaries in engineering documentation. A copied placeholder that fails at startup is worse than a shorter honest example.

Watch memory during replica startup, not only at steady state. Rolling updates can briefly overlap old and new replicas. A model that consumes almost all GPU memory per replica may prevent a new replica from becoming healthy before the old one drains. That makes deployment strategy and spare capacity part of model sizing.

### The concurrency number is a limit, not a capacity claim

`max_ongoing_requests` caps how many requests a replica may have assigned to it without finishing. The current default is 5, changed from 100 in Ray 2.32.0 according to the [deployment configuration guide](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html). Do not treat five as a universal optimum. An I/O-heavy handler may benefit from a larger value because many calls are waiting. A memory-hungry GPU handler may need a lower value because each request consumes KV cache or temporary tensors. Record p50, p95, p99, throughput, and memory as you sweep this limit under representative traffic.

Also distinguish placement from execution. Ray can reserve one CPU for an actor, while a native numerical library starts many threads unless you configure it. That can oversubscribe a node. Conversely, reserving more CPUs does not automatically vectorize your model. Check the library's thread settings and benchmark the actual combination.

## 5. Backpressure before autoscaling

**Rule of thumb: define what happens at overload before counting on new replicas.** A service that accepts requests faster than it can finish them is building a latency debt.

![Bounded caller queues reject excess traffic instead of allowing tail latency to grow without limit](/imgs/blogs/ray-serve-production-engineering-4.webp)

A request can be ongoing on a replica or waiting at a caller such as the proxy or another deployment handle. `max_ongoing_requests` controls the former. `max_queued_requests` controls how many the caller will queue for a deployment. When the caller queue is full, Serve raises `BackPressureError` for handles and returns HTTP 503 by default for HTTP traffic. The [production best-practices guide](https://docs.ray.io/en/latest/serve/production-guide/best-practices.html) shows these parameters and explains how to customize the response to HTTP 429 with `Retry-After` when intentional load shedding should be distinct from server failures.

Suppose one replica handles 20 requests per second and traffic jumps to 100 per second for five seconds. Ignoring concurrency and assuming constant service time, approximately 400 excess requests arrive during that burst. If all wait, even a perfectly healthy model needs roughly 20 seconds to drain them after the burst. A 1-second endpoint SLO is already lost. Adding a replica after ten seconds cannot undo the time those early requests spent waiting. This is a worked queueing example, not a Ray benchmark.

A bounded queue makes that debt visible. You choose the queue budget from an end-to-end latency target. If a request may wait at most 200 ms and a replica finishes a unit of work every 50 ms, a queue much deeper than four requests per serial capacity lane already threatens the budget. Real services have variable service times and parallel execution, so validate with a load test. The arithmetic gives a starting intuition: queue capacity is a latency decision, not just a memory setting.

```python
from ray import serve


@serve.deployment(
    max_ongoing_requests=4,
    max_queued_requests=32,
)
class BoundedService:
    async def __call__(self) -> str:
        return "ready"
```

This is a runnable declaration, though you still need to bind and run it. For an external API, also set client timeouts and a limited retry policy. Retry on errors that are likely transient and safe to repeat. Use exponential backoff with jitter, cap attempts, and respect `Retry-After` when provided. A user-visible generation request may not be safe to replay blindly after partial streaming output. The server cannot turn an ambiguous client retry into exactly-once execution for you.

| Overload policy | Client experience | Operational consequence |
| --- | --- | --- |
| Unlimited practical queue | Late success or timeout | p99 rises and memory pressure can grow |
| Bounded queue, 503 | Fast explicit failure | Availability alerts may include deliberate shedding |
| Bounded queue, 429 plus `Retry-After` | Fast rate-limit signal | Client and gateway policies can treat it separately |
| Aggressive automatic retry | Sometimes hides a transient | Can multiply traffic during the very incident it tries to fix |

A second-order failure occurs when every stage has a large queue. A top-level request can wait at retrieval, then at reranking, then at generation. Each queue looks locally acceptable while their sum breaks the user SLO. Budget waiting time across the entire graph, and propagate deadlines so downstream work can stop when the client has already given up.

## 6. Autoscale the work, then the cluster

**Rule of thumb: measure one replica first, then set the scaling target from its latency curve.** The autoscaler cannot infer your SLO from a decorator.

Ray Serve can use a fixed `num_replicas` or autoscaling. The simple `num_replicas="auto"` setting enables documented defaults. In Ray 2.58.0, the [autoscaling guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html) shows `target_ongoing_requests=2`, `max_ongoing_requests=5`, `min_replicas=1`, and `max_replicas=100` for that preset. A manually supplied `autoscaling_config` has its own base defaults; notably, leaving `max_replicas` at one would prevent any scale-out. Always state your bounds explicitly in production rather than assuming a preset fits your cost and latency objectives.

The Serve autoscaler responds to request demand by changing the desired number of deployment replicas. If Ray has no free resources to place those actors, the Ray autoscaler can request more nodes, assuming cluster autoscaling is configured. These are two separate loops. The first cannot conjure GPUs. The second cannot shorten model initialization once a GPU arrives. This relationship is described in the [Serve autoscaling guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html).

<figure class="blog-anim">
<svg viewBox="0 0 920 210" role="img" aria-label="Traffic rises, Serve requests more replicas, Ray adds a node if resources are insufficient, and the new replica becomes ready" style="width:100%;height:auto;max-width:980px">
<style>
.rsas-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.rsas-active{fill:var(--accent,#6366f1);opacity:.18}.rsas-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.rsas-line{stroke:var(--border,#9ca3af);stroke-width:3}.rsas-dot{fill:var(--accent,#6366f1)}@keyframes rsas-step{0%,18%{transform:translateX(0)}25%,43%{transform:translateX(210px)}50%,68%{transform:translateX(420px)}75%,93%{transform:translateX(630px)}100%{transform:translateX(0)}}.rsas-move{animation:rsas-step 8s steps(1,end) infinite}@media (prefers-reduced-motion:reduce){.rsas-move{animation:none}}
</style>
<line class="rsas-line" x1="145" y1="102" x2="775" y2="102"/>
<rect class="rsas-box" x="15" y="58" width="175" height="90" rx="12"/><rect class="rsas-box" x="225" y="58" width="175" height="90" rx="12"/><rect class="rsas-box" x="435" y="58" width="175" height="90" rx="12"/><rect class="rsas-box" x="645" y="58" width="175" height="90" rx="12"/>
<rect class="rsas-active rsas-move" x="20" y="63" width="165" height="80" rx="10"/>
<text class="rsas-label" x="102" y="91">Traffic rises</text><text class="rsas-label" x="102" y="115">queue grows</text><text class="rsas-label" x="312" y="91">Serve asks for</text><text class="rsas-label" x="312" y="115">more replicas</text><text class="rsas-label" x="522" y="91">Ray adds node</text><text class="rsas-label" x="522" y="115">if needed</text><text class="rsas-label" x="732" y="91">Replica loads</text><text class="rsas-label" x="732" y="115">and serves</text>
<text class="rsas-label" x="460" y="188">Capacity arrives after scheduling and startup</text>
</svg>
<figcaption>Traffic can rise immediately; replica placement and model startup take time before useful capacity appears.</figcaption>
</figure>

Start with a load test on one replica using representative payload sizes. Increase concurrent clients until throughput stops rising or p99 crosses the SLO. That point helps choose `target_ongoing_requests`. Then choose `max_ongoing_requests` high enough to keep the replica productive but low enough to avoid piling too much work into it. The [advanced autoscaling guide](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html) suggests setting the maximum ongoing value roughly 20 to 50 percent above the target as an initial heuristic, while warning that lightweight requests may need a higher value. It is a starting point, not a substitute for measurement.

```yaml
applications:
  - name: rag
    route_prefix: /
    import_path: rag_app:app
    deployments:
      - name: Retriever
        autoscaling_config:
          min_replicas: 2
          max_replicas: 8
          target_ongoing_requests: 2
        max_ongoing_requests: 3
        max_queued_requests: 32
      - name: Answerer
        autoscaling_config:
          min_replicas: 2
          max_replicas: 4
          target_ongoing_requests: 2
        max_ongoing_requests: 3
        max_queued_requests: 32
```

These values are illustrative. In particular, a toy retriever will have a very different capacity curve from a real vector database client. The YAML is a Serve config overlay on the `rag_app.py` import path. The actual deployment name must match the class name or explicit deployment name. For a production service, benchmark the real stages and calculate a peak bound from measured capacity. If a replica sustainably handles 30 QPS at the required p99 and expected peak is 180 QPS, six replicas are a theoretical minimum before headroom, uneven routing, updates, and failures. A starting maximum of eight or nine is more realistic, subject to node and cost limits.

Scale-to-zero sets `min_replicas=0` and saves idle resources, but the first request waits for a replica, and possibly a node, to start. That can be acceptable for internal or low-frequency workloads. It is often incompatible with a tight interactive p99. A nonzero minimum buys warm capacity. The right setting is an economic decision expressed through a latency SLO, an idle-cost budget, and measured startup time.

Finally, test the whole chain. A desired replica count of eight is not eight running replicas. Check `serve status`, Ray available resources, Kubernetes node autoscaling if applicable, image pull, model loading, and readiness. Raising `max_replicas` does nothing if a GPU request cannot be placed.

## 7. Batch and allocate resources deliberately

**Rule of thumb: tune batching against the endpoint SLO, not against GPU utilization alone.** The model may be faster per item in a batch while the first item waits longer for companions.

![Batching trades formation wait against throughput and can worsen tail latency](/imgs/blogs/ray-serve-production-engineering-6.webp)

Ray Serve's `@serve.batch` groups calls to a decorated async method. The method receives a list and must return a list of equal length so Serve can hand each result back to the corresponding caller. `max_batch_size` caps the batch, and `batch_wait_timeout_s` determines how long Serve waits after the first request arrives before dispatching an incomplete batch. The [dynamic batching guide](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html) documents the behavior and the current defaults. Write the method as an actual vectorized computation or one engine call over several inputs; wrapping a Python `for` loop around individual model calls may produce little benefit.

```python
from ray import serve


@serve.deployment
class BatchedSquares:
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.01)
    async def predict(self, values: list[int]) -> list[int]:
        return [value * value for value in values]

    async def __call__(self, value: int) -> int:
        return await self.predict(value)


app = BatchedSquares.bind()
```

Save that as `batch_app.py`, run `serve run batch_app:app`, and call it through a deployment handle or adapt the ingress to parse HTTP. The arithmetic is intentionally cheap, so it demonstrates decorator semantics rather than a speedup. For a real model, replace the list comprehension with a batch-capable library call. Never promise a throughput gain from the decorator alone.

Here is a worked latency budget. Suppose an endpoint has a 150 ms p99 target. In a representative benchmark, ingress plus network plus downstream work already consume 25 ms and the model batch takes 100 ms. That leaves at most 25 ms for queueing and batch formation combined, with no margin for variation. A `batch_wait_timeout_s` of 0.1 seconds would be incompatible with the target even if it improves GPU utilization. A 5 ms or 10 ms wait may be a reasonable experiment, but only a load test at both low and high arrival rates can tell you. These numbers are illustrative, not published Ray performance data.

Arrival rate is crucial. At high QPS, eight requests may arrive within a few milliseconds and the batch fills quickly. At low QPS, the first request can wait nearly the entire timeout and still be processed alone. This makes batching performance load-dependent. Report p99 across a traffic distribution, including quiet periods, rather than only at peak throughput. For a customer-facing chatbot, time to first token is often more visible than total tokens per second. For offline embedding, throughput may dominate. The same batch parameters should not be copied between them.

Batch composition also affects memory. Eight short prompts and eight long prompts are not equivalent. The [batching guide](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html) supports a `batch_size_fn` for workload-specific limits such as total tokens, graph nodes, or pixels. Use a count-based limit only when requests are reasonably uniform. If a few large inputs can OOM a replica, constrain by the resource they actually consume.

| Parameter | Raise it when | What can go wrong |
| --- | --- | --- |
| `max_batch_size` | Larger batches measurably improve work per GPU call | More VRAM, longer batch service, head-of-line blocking |
| `batch_wait_timeout_s` | Arrival rate is low and batch efficiency matters more than latency | First request waits too long |
| `max_concurrent_batches` | Engine safely overlaps batches and memory permits it | Contention or OOM |
| `max_ongoing_requests` | Replica can productively hold more concurrent requests | Queue moves into the replica |
| GPU fraction per replica | Small models can share one GPU with measured headroom | Scheduler reservation is not VRAM isolation |

Resource allocation deserves its own benchmark matrix. Start with one replica on a dedicated device and record throughput, p99, peak memory, and startup time. Then test two replicas sharing the same device if the models are small enough. Compare the total cost per successful request at the required p99, not the number of actors you managed to schedule. The [Ray Serve resource guide](https://docs.ray.io/en/latest/serve/resource-allocation.html) explains how actor resource requests control placement. They do not promise hard GPU memory partitioning.

The second-order effect is batch boundaries across stages. An upstream stage that emits groups of eight into a downstream stage tuned for groups of six can cause the downstream stage to process one full batch and leave a partial batch waiting. The [batching documentation](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html) calls out this interaction. When you chain batched deployments, benchmark the chain, not just each stage in isolation.

## 8. Deploy and update without guesswork

**Rule of thumb: keep application code, Serve settings, and cluster settings in clearly named layers.** A configuration change is safer when you know which process will restart and which will not.

![Python code Serve config and RayService govern different layers of a production deployment](/imgs/blogs/ray-serve-production-engineering-7.webp)

During development, `serve run module:app` is a direct feedback loop. For production, the [Ray production guide](https://docs.ray.io/en/latest/serve/production-guide/index.html) recommends Kubernetes with the KubeRay `RayService` custom resource. The [Serve config guide](https://docs.ray.io/en/latest/serve/production-guide/config.html) describes the Serve config as the recommended way to define applications, routes, imports, and deployment overrides. On a VM or an existing Ray cluster, the Serve CLI remains an option.

A practical sequence is:

```bash
python -m pip install 'ray[serve]==2.58.0'
serve run rag_app:app
serve build rag_app:app -o serve_config.yaml
serve deploy serve_config.yaml
serve status
serve config
```

The generated config is a starting artifact. Review the application name, route prefix, deployment names, resource reservations, scaling bounds, and HTTP settings before using it. `serve status` reports application and deployment health. `serve config` shows the latest received goal configuration. The production [best-practices guide](https://docs.ray.io/en/latest/serve/production-guide/best-practices.html) recommends these commands for exactly that development-to-deployment flow. A successful CLI exit is not proof that all replicas are healthy or that requests meet the SLO; query status and send test traffic.

In production, build the application code and dependencies into a versioned Docker image. The docs recommend a custom image over fetching code through a runtime environment for production. This improves reproducibility and avoids treating a mutable repository branch as a deployment artifact. Pin Ray, Python, model library versions, and any engine dependency. Record the image digest and the model artifact version separately. A code rollback is less useful if the model weights or tokenizer silently changed.

A Serve config can override some deployment settings without editing Python code. That is useful for `num_replicas`, autoscaling values, and `user_config`. Do not assume every field is hot-updatable. The [config reference](https://docs.ray.io/en/latest/serve/production-guide/config.html) notes that HTTP options are global and cannot be updated at runtime. The operational question is not simply “Can I change this YAML?” It is “What gets restarted, and do I have enough spare capacity while old and new replicas overlap?” Test the exact update path in staging.

KubeRay's `RayService` takes responsibility for reconciling the Serve application and Ray cluster on Kubernetes, including health reporting, recovery, and upgrades. Kubernetes then adds its own capacity and scheduling layer. A deployment can be healthy in Serve while external ingress routing is wrong. Conversely, a Kubernetes pod can be running while the model has not loaded and the Serve replica is not ready. Check both levels. If Ray cluster autoscaling requests a GPU node, your cloud or Kubernetes node autoscaler must still provision it. The [Kubernetes deployment guide](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html) covers the moving pieces.

| Layer | Example ownership | Change to verify |
| --- | --- | --- |
| Python application | Handler logic, model initialization, deployment graph | Functionality and model compatibility |
| Serve config | Application route, scaling, queue and concurrency limits | Replica rollout, status, and SLO |
| Ray cluster | Node types, actor placement, available CPU/GPU | Pending actors and node provisioning |
| RayService and Kubernetes | Cluster lifecycle, images, services, ingress | Readiness, upgrades, external reachability |
| Client and gateway | Timeouts, retry, authentication, rate limit | End-to-end behavior and failure amplification |

For safe updates, keep at least one known-good revision available during validation. Send a canary request that exercises each stage, not merely `/` returning a static health string. Test a request whose model initialization path and tokenization path resemble production. Watch error rate, p99, queue depth, GPU memory, and replica startup during the update. Roll back based on measured degradation rather than waiting for a complete outage.

The second-order trap is a warm-up mismatch. A new replica may pass a basic process health check before a model's first real inference has compiled kernels, populated caches, or loaded secondary assets. If the first user request pays that cost, a rolling update can create a latency spike even though all components look healthy. Warm up with representative input and measure first-request latency separately from steady state.

## 9. Observe the service and debug p99

**Rule of thumb: follow the request's waiting time, then inspect compute.** Adding CPU or GPU because p99 rose is a guess until you know where time accumulated.

![A high p99 branches into queue replica and cluster resource diagnoses](/imgs/blogs/ray-serve-production-engineering-8.webp)

The [monitoring guide](https://docs.ray.io/en/latest/serve/monitoring.html) covers Serve status, logs, metrics, and dashboard views. At minimum, graph request rate and status by route, end-to-end latency, deployment processing latency, ongoing requests, queue behavior, running versus desired replicas, and resource utilization. For an LLM, add time to first token, tokens per second, prompt and output lengths, and engine queue depth. A chart without deployment and model labels hides the stage that needs attention.

A useful diagnostic procedure begins with a load test that records the offered request rate. If offered rate rises while completed rate stays flat, identify whether rejections rise or requests wait. If queue length rises while replica processing time is stable, the stage has insufficient capacity or a limit that is too tight. If processing time rises at the same concurrency, the replica itself is slowing down: inspect CPU contention, GPU memory pressure, batch shape, external dependencies, and input size. If desired replicas rise but running replicas do not, inspect Ray resource availability and node provisioning. If running replicas rise but p99 does not recover, revisit the bottleneck assumption and the load balancer path.

Consider a hypothetical endpoint whose p99 jumps from 300 ms to 2 seconds after a new document corpus is indexed. A naive response is to double generation replicas because generation is the most expensive stage. But suppose traces show retrieval rising from 20 ms to 700 ms while generation remains stable. New generation GPUs will sit idle while requests wait on the vector store. The right investigation is retrieval query shape, index memory, network, and top-k size. Independent deployments make this distinction visible if you instrument them; they do not diagnose it automatically.

Logs answer a different question from metrics. Metrics show the population-level symptom. Logs explain one replica's exception, model load failure, or request ID. Trace context ties the user's HTTP request to internal handle calls. If the service calls an external vector database or remote LLM, propagate a correlation ID across that boundary as well. Without it, a 503 on the endpoint and a timeout in the vector store may be nearby in time but hard to connect confidently.

```bash
serve status
serve config
ray status
curl -fsS http://127.0.0.1:8000/-/healthz
```

Run these from the appropriate cluster environment and use your actual endpoint address. The CLI tells you whether Serve received the intended config and whether Ray has resources; the health probe tests ingress reachability. It does not replace a representative application request. On Kubernetes, inspect `RayService`, pod events, and Serve logs as well. The [Kubernetes guide](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html) points to Serve controller and deployment logs under the Ray session log directory.

| Symptom | Evidence to gather | Likely next experiment |
| --- | --- | --- |
| Queue grows, processing time steady | Offered versus completed QPS, ongoing requests | Add warm capacity or shed earlier |
| Processing time grows with load | CPU/GPU saturation, input length, batch shape | Profile replica and lower unsafe concurrency |
| Desired replicas exceed running replicas | Ray resource status, pending actors, node events | Fix placement or provision capacity |
| Errors spike after update | Image and model versions, startup logs | Roll back or fix initialization |
| Low-load p99 worsens after batching | Batch wait and actual batch size | Shorten wait or disable batching for that path |
| Many 429/503 responses during burst | Queue limit and client retry rate | Tune admission and client backoff together |

The second-order concern is cardinality. Label metrics with stable deployment and model identifiers, but do not create a metric series per user ID, request ID, or arbitrary prompt. Put high-cardinality identifiers in logs or traces. Otherwise the monitoring system itself can become a production bottleneck.

## 10. Serve LLM and the serving stack

**Rule of thumb: separate the engine that computes tokens from the serving layer that composes and operates endpoints.** The distinction prevents both underbuilding and overbuilding.

![Ray Serve owns application orchestration while the inference engine owns model execution](/imgs/blogs/ray-serve-production-engineering-9.webp)

Ray Serve LLM builds on Serve primitives for distributed LLM workloads. Its [architecture overview](https://docs.ray.io/en/latest/serve/llm/architecture/overview.html) describes an LLM server deployment that manages an engine instance and an ingress layer that exposes model-facing APIs. It discusses horizontal scaling, multi-node models, and more specialized patterns such as disaggregated prefill and decode. These capabilities matter when a single engine process is no longer the whole service. They do not remove the need to understand engine-level batching, KV cache, quantization, or prompt length.

For a single model on one node, start with the engine's own server and benchmark it. If it meets the SLO and you do not need Serve's composition or cluster operations, that may be sufficient. Add Serve when you need application-specific preprocessing, multiple model deployments with independent scaling, routing across a Ray cluster, or Serve LLM's distributed patterns. Keep the decision empirical. A new control plane should be justified by an operational requirement.

A RAG endpoint is a good example of complementarity. Serve can own the embedding, retrieval, reranking, and generation graph. The generation deployment may use vLLM to schedule token generation efficiently. vLLM's internal scheduler and Serve's deployment autoscaler answer different questions. One decides how an engine uses its allocated device; the other decides how many engine-bearing replicas should exist and where they are placed. Tuning only one side leaves capacity or latency on the table.

| Layer | Main question | Examples |
| --- | --- | --- |
| Application composition | Which stage calls which, and what scales independently? | Ray Serve deployments and handles |
| LLM service interface | Which model receives the request and how is it streamed? | Ray Serve LLM ingress and routing |
| Inference engine | How are model weights, attention, and tokens executed? | vLLM or another supported engine |
| Cluster scheduler | Which node has the CPU/GPU resources? | Ray scheduler and autoscaler |
| Infrastructure | Which machines and network endpoints exist? | Kubernetes, cloud nodes, load balancer |

For distributed LLM inference, think about the replica as a larger unit than one Python process. An engine may require multiple GPUs or nodes before one useful model instance is ready. Autoscaling therefore has a step size and a startup cost. If a replica spans several nodes, the whole placement group must fit. A cluster with many free GPUs scattered across unsuitable nodes can still fail to place the desired topology. The [Serve LLM architecture](https://docs.ray.io/en/latest/serve/llm/architecture/overview.html) and [multi-node vLLM guide on this site](/blog/machine-learning/inference-frameworks/running-vllm-distributed-in-production) provide the next level of detail.

Routing policy matters at this layer too. The [Serve LLM routing guide](https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html) distinguishes model selection at ingress from replica selection within a deployment. Its default replica policy uses power of two choices. Prefix-aware routing can favor replicas whose KV cache already contains relevant prefixes, subject to load balance. That is useful when shared prompts create substantial cache reuse; it is not a free improvement for workloads with unrelated prompts. Measure cache-hit rate and p99 together before switching routing policy.

### Many small models: multiplexing changes the memory equation

Some products serve hundreds of related models with the same input shape but different weights. Giving every model a dedicated replica can waste memory when most models receive only occasional traffic. Ray Serve's [model multiplexing guide](https://docs.ray.io/en/latest/serve/model-multiplexing.html) describes another pattern: a pool of replicas loads models on demand and routes later requests toward replicas that already have the requested model. `@serve.multiplexed(max_num_models_per_replica=...)` bounds how many models one replica holds. When the bound is exceeded, Serve evicts the least recently used model from that replica.

This is a cache, so it has a hit-rate problem. If ten models are requested repeatedly and the replica pool can hold all ten, most requests may avoid model-load time after warm-up. If a hundred equally popular models compete for ten slots, loads and evictions can dominate p99. Before adopting multiplexing, measure model size, load time, request frequency by model ID, and memory headroom. A single global request rate hides a long tail of cold models. Plot latency by warm hit versus cold load, then decide whether to reserve dedicated capacity for the hottest models and multiplex the tail.

The routing identifier is part of the request contract. The current guide uses a `serve_multiplexed_model_id` HTTP header or a deployment handle option to identify the target model. Validate and authorize that ID. A caller must not be able to select another tenant's model merely by changing a header. If the application loads artifacts from remote storage, make the artifact path and version immutable, and reject unknown IDs rather than concatenating untrusted strings into storage paths. A cache eviction must release the underlying model resources, not just remove a Python dictionary entry.

Multiplexing also interacts with batching. Serve can separate batches by model ID, because one model call cannot normally mix weights from two different models. That preserves correctness but can reduce batch fill for a long tail of low-traffic IDs. A high aggregate QPS does not imply a high per-model QPS. Test both the common model and sparse models. If the long tail has strict first-request latency requirements, multiplexing may save GPU memory while failing the product SLO. A dedicated replica or pre-warming policy can be preferable for those models.

This is a good example of why a serving feature is a tradeoff rather than a checkbox. Dedicated replicas spend memory to buy predictable latency. Multiplexed replicas spend occasional load time to share memory. The right choice depends on the popularity distribution, artifact load cost, isolation needs, and the service-level objective for cold requests.

### Derive a capacity plan from one measured replica

A capacity plan begins with the request distribution, not with a replica count. Run the service under representative input sizes and record a curve, not one headline QPS number. At each offered rate, record completed QPS, rejection rate, p50 and p99, CPU and GPU use, memory, and the number of ongoing and queued requests. The useful point is the highest sustained rate that still meets the latency objective with stable memory and no growing queue. Call that measured rate $q_{safe}$ requests per second per replica. If expected peak traffic is $q_{peak}$, a first estimate is $\lceil q_{peak}/q_{safe}\rceil$ replicas. That estimate is an engineering calculation, not the Serve autoscaler's implementation rule.

Suppose a real load test, not this article, found that one replica could sustain 24 QPS at p99 below 500 ms. A forecast peak of 120 QPS would require at least five replicas under ideal balance. Five is an unsafe production maximum: one replica may be updating, inputs vary, and the peak forecast can be wrong. A maximum of seven or eight is a reasonable test candidate if the cluster can place them and the budget allows it. A minimum of two might cover the overnight baseline. You would then replay a burst from baseline to peak and observe whether scale-out arrives before the latency objective is violated. If it does not, raise warm capacity or reduce startup time. Merely changing the target value cannot erase cold-start physics.

Traffic shape also matters. A five-minute average of 100 QPS could mean a smooth 100 QPS, or a ten-second spike at 1,000 QPS followed by quiet. Those patterns produce different queues and autoscaler responses even though their averages match. Replay realistic arrival processes, including synchronized clients, slow users, long prompts, and retries. If your business has a predictable opening bell or scheduled batch partner, scale ahead of it or reserve headroom. Reactive scaling is strongest against sustained demand, not instantaneous bursts shorter than a model's startup time. Separate traffic forecasts for weekdays, scheduled jobs, and incident recovery. The minimum replica count can be adjusted through a reviewed configuration change before a known event, then returned to baseline when the event ends. Record that operation in the same change log as code releases so a temporary capacity increase does not become permanent spend. If scaling is automated externally, be clear which controller owns the desired count; competing controllers can repeatedly overwrite each other's decisions.

Capacity should also be checked per availability zone or failure domain. Eight replicas on one node are eight processes but only one node's worth of resilience. Two replicas on two nodes can survive a different class of failure. Placement topology matters as much as count when a service promises availability. Test the actual loss of a node, not just the termination of an actor in an otherwise healthy node. If one zone has all the warm GPU capacity, a regional or zone-level disruption can reveal that the nominal replica count never represented independent capacity.

The end-to-end timeout should be greater than normal service time but smaller than the point where the result has no user value. Put a separate timeout around external dependencies. A retrieval call that hangs for eight seconds should not consume an interactive generation budget of one second. Timeouts, cancellation, and retries must be designed together: a client giving up does not necessarily imply that downstream work instantly stops. Test cancellation behavior with an intentionally slow handler and watch whether ongoing work clears. The Serve [config reference](https://docs.ray.io/en/latest/serve/production-guide/config.html) documents the HTTP `request_timeout_s` setting and notes that there is no default request timeout. That makes an explicit application and gateway deadline especially important.

A simple load-test report can fit in one table. Fill it with your own measurements rather than copying the example values below:

| Test condition | Offered QPS | Completed QPS | p99 | Queue trend | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| One replica, steady | 15 | 15 | 220 ms | Flat | Inside budget |
| One replica, near knee | 24 | 24 | 480 ms | Flat | Candidate safe capacity |
| One replica, overload | 35 | 24 | 2.1 s | Growing | Shed or scale |
| Six replicas, planned peak | 120 | 120 | 490 ms | Flat | Check failure headroom |

Every number in this table is illustrative. The decision procedure is the artifact. The point where throughput flattens and latency rises sharply is the knee of the curve. Set autoscaling and admission so ordinary traffic stays away from it.

### Treat failure recovery as a capacity event

Serve can replace failed replica actors, and the controller can recover components, but replacement takes time and transient in-flight work can be lost. The [architecture guide](https://docs.ray.io/en/latest/serve/architecture.html) distinguishes durable control information from transient queues and connections. Do not read “fault tolerant” as “every request completes exactly once.” Clients need timeouts and safe retry rules; the service needs idempotency where a repeated operation could have side effects.

The most useful failure drill is to remove one replica while the service is at its normal high load. If the remaining replicas can absorb its traffic without breaking p99, you have N-1 capacity for that event. If they cannot, your steady-state utilization is too high for the availability objective. Repeat with one node loss, because a node may host several replicas or an entire multi-GPU engine unit. In a GPU cluster, placement constraints can make nominally free GPU capacity unusable for the replica shape you need.

Also decide what degraded service means. A RAG system might return retrieved evidence without generation when the generator is unavailable, or it might return an explicit temporary error. Either choice can be correct for a particular product. Document it and test it. Silently returning a low-quality answer while reporting HTTP 200 can hide an outage from availability metrics and violate user expectations. Graceful degradation is a product contract, not merely an exception handler.

### Separate production security from Serve's internal convenience

The convenience of a local `serve run` endpoint is useful for development. A production endpoint also needs authentication, authorization, transport security, input limits, secrets management, and an exposure policy for dashboards and control APIs. Put an authenticated gateway or ingress in front of public traffic and keep cluster control interfaces on trusted networks. The serving graph should still validate request payloads and bound expensive input dimensions, such as prompt length, retrieved document count, image resolution, or maximum generated tokens. An attacker does not need high QPS to exhaust a GPU if one request can allocate excessive memory.

Reserve sensitive credentials for the deployment that actually uses them. A retrieval stage may need database credentials; a generation stage may not. Avoid baking secrets into images or logging raw prompts by default. Set retention and redaction policies for request logs. If tenants share a Serve application, define per-tenant admission and observability at the gateway or application layer. A global queue limit protects the cluster but does not guarantee fairness between one noisy tenant and everyone else.

These concerns are not a claim that Ray Serve lacks security controls. They identify the boundary of what this article's sample application demonstrates. The toy RAG app is local and unauthenticated. Before exposing it externally, specify the gateway, network, credential, and data-handling controls that your environment requires.

### Run a load test that can answer a design question

A useful load test is an experiment with one independent variable. If you change replica count, batch size, model version, and prompt distribution together, the result cannot explain which change helped. Begin with a baseline that captures code revision, Ray version, image digest, model artifact, node type, replica count, concurrency settings, and the exact request generator. Use a separate client machine or verify that the client itself is not saturated. Warm the service before the steady-state phase, then run a cold-start phase separately. Report both; neither is a substitute for the other.

Use an open-loop arrival generator when you want to model independently arriving users. A closed-loop test, where each simulated user waits for a response before sending another request, automatically reduces offered load when the service slows. That can conceal overload. For example, 100 closed-loop users may produce 100 QPS when responses take one second, but only 50 QPS when responses take two seconds. The graph may misleadingly show stable errors because the client has backed off without intending to. Open-loop tests must still cap outstanding work so the load generator does not exhaust itself. Record offered, admitted, completed, rejected, and timed-out rates separately.

Test at least four regimes: idle or low load, normal steady load, peak sustained load, and a short burst above peak. The low-load run reveals batch formation delay and cold paths. The steady run gives an efficiency baseline. The sustained peak shows whether queues remain stable. The burst tests admission and scale-up. Then remove a replica or node during a peak run to check failure headroom. Stop the experiment when a safety threshold is reached; deliberately sending unlimited requests into an unbounded queue tells you little beyond the fact that queues grow.

For LLMs, fix or stratify prompt lengths and output lengths. A 200-token prompt asking for 20 output tokens is not the same workload as an 8,000-token prompt asking for 1,000. Report time to first token, time between output tokens, total completion time, token throughput, and request success rate. If a benchmark reports only requests per second, it can reward short outputs that do not resemble production. If it reports only tokens per second, it may hide a poor first-token experience. The [Serve LLM observability guide](https://docs.ray.io/en/latest/serve/llm/user-guides/observability.html) describes service and engine metrics that can support this view.

One more rule prevents false confidence: compare at equal quality and equal SLO. A system that drops hard requests or truncates outputs may appear cheaper per request. Report rejected requests and output lengths beside resource cost. The useful efficiency measure is cost per *successful request meeting the latency target*, with the workload and output contract held constant. That quantity can move in the opposite direction from raw GPU utilization.

### Make cost a constraint, not a separate afterthought

Serve's per-deployment scaling helps cost only when the resource profile is known. Let $c_i$ be the cost per hour of one replica of stage $i$, and $r_i(t)$ its running replica count at time $t$. A simple infrastructure estimate is $C=\int \sum_i c_i r_i(t)\,dt$. This is an **explanatory cost model** for capacity planning, not an equation supplied by Ray. It ignores shared cluster overhead, network, storage, and reserved idle nodes until you add them explicitly. Its value is that it forces you to ask whether a deployment really scales down and whether the cluster can release the underlying machine.

Consider a CPU retriever and GPU generator. If each GPU replica costs far more per hour than several CPU replicas, separating the stages can avoid duplicating the retriever on every GPU process. But the saving materializes only if the GPU replicas can be reduced without leaving a paid GPU node running idle. On Kubernetes, node bin-packing, GPU type, and minimum node pools determine the bill. A fractional GPU reservation does not by itself halve the cloud bill. Track actual node-hours and utilization, not just Ray's logical resource units.

There is also a latency cost to extreme efficiency. A minimum of zero saves idle compute but pays cold-start time. A high batching timeout improves device use but makes quiet requests wait. Running GPUs near full utilization leaves less room for a burst or a failed node. Put these choices on one chart: p99, error rate, and cost per successful request at the same offered load. Choose a configuration on the acceptable frontier for your product, not the one with the prettiest utilization line.

### Write a short incident runbook before the first incident

A runbook should start with observable facts and end with one reversible action. Begin with the timestamp, affected route, application version, offered request rate, error classes, and p99. Check whether the incident began with a rollout, a traffic change, or a dependency change. Then inspect queues, per-stage processing time, desired and running replicas, and node resources. The first decision is whether to shed load, restore a known-good version, or add capacity. If the queue is rising rapidly, shedding may protect current users while the deeper fix is prepared.

Record what each command is expected to show. `serve status` tells you application and deployment state. `serve config` shows the received goal state. `ray status` exposes cluster resources and pending demands. Kubernetes events explain failed pod scheduling or image pulls. A representative request checks the whole graph. A single green health endpoint does not prove that retrieval credentials work or that the generator has loaded weights. Include links to dashboards and logs by deployment so the on-call engineer does not have to discover them during an outage.

Use rollback when a revision changed code, configuration, or model artifacts and the old revision is known to satisfy the SLO. Use capacity changes when the workload grew but code and per-replica processing are stable. Use dependency mitigation when a downstream service is slow. Each action should have an expected signal and a stop condition. For example: “Raise the warm generator floor from two to four; within two model startup intervals, running replicas should reach four and queue length should begin falling.” If running replicas remain at two, the hypothesis was wrong or placement is blocked. That is evidence to change direction, not a reason to raise the requested count again.

Finally, capture the incident after recovery. Save the traffic shape, relevant traces, settings, root cause, and the one alert that would have shortened diagnosis. Turn the failure into a repeatable drill. A serving stack becomes reliable through tested behavior under overload and partial failure, not through a long list of configuration parameters.

## 11. Six incident drills to practice before production

These are **worked scenarios**, not claims about incidents observed at a named company. Each uses illustrative numbers so the diagnosis is concrete. Reproduce the shape with your own load test before applying the setting.

### 11.1. The queue that looked like a slow model

At 09:00, an API's p99 rises from 450 ms to 4 seconds. GPU utilization looks high, so the first hypothesis is that a new model revision became slower. A sample trace shows generation itself still takes about 300 ms. Most of the time is spent before the generator starts. Request rate doubled during a partner's batch import, and the caller queue kept growing because admission was effectively unbounded for the burst. The model did not slow down; waiting work accumulated faster than the current replicas could retire it.

The immediate fix is to enforce a queue limit, return an explicit overload response, and make the partner client back off. Then run a load test to decide whether a larger warm replica floor or a faster scale-up path is economical. Setting a 30-second timeout would merely make the request wait longer and retain resources. Raising `max_ongoing_requests` without checking engine capacity could move the backlog inside replicas, where it is harder to shed and where KV memory may explode. The lesson is to graph processing latency and waiting time separately. A slow response is not proof of slow compute.

### 11.2. The autoscaler wanted eight replicas but ran two

A deployment is configured with `max_replicas=8`, and a traffic burst clearly exceeds two replicas' capacity. `serve status` shows scale-out intent, but only two replicas are running. The first hypothesis is a Serve autoscaler bug. `ray status` shows that no node has enough free GPU resources for a third replica. On Kubernetes, a GPU node is still provisioning and pulling a multi-gigabyte image. Serve made the correct request; cluster capacity did not arrive in time.

The fix depends on the SLO. Keep more warm capacity for predictable peaks, shrink image and model startup work, or improve node provisioning. Do not claim that changing the autoscaling target alone solves the placement delay. In a test, measure the four intervals separately: demand detection, node arrival, actor placement, and model readiness. If the last interval dominates, preloading or warm-up can matter more than the autoscaler control interval. The lesson is that an autoscaling policy describes desired replicas, while the infrastructure determines how fast those replicas become useful.

### 11.3. A larger batch raised throughput and broke p99

A reranker moves from a batch size of four to 32. At a synthetic peak of 400 requests per second, GPU throughput improves and average device utilization rises. The rollout looks successful on a throughput dashboard. At night, traffic falls to eight requests per second. The first request in each batch now waits for the batch timeout because the batch rarely fills. A 100 ms wait consumes most of a 250 ms p99 target before inference begins.

The fix is to shorten the wait, reduce the maximum batch size, or separate a latency-sensitive route from a throughput-oriented route. Use a workload-specific batch size function if candidate lengths vary widely. Measure at low, medium, and peak arrival rates. The root cause is not that batching failed; it optimized the wrong operating point. A service must meet its SLO across the traffic it actually sees. Report batch-size distribution and formation wait alongside GPU utilization, because utilization alone rewards latency debt.

### 11.4. The async handler blocked everything around it

An engineer changes a synchronous handler to `async def` and raises `max_ongoing_requests` from five to 100, expecting more parallel work. The handler still calls a blocking Python client and performs CPU-heavy text processing before the first `await`. Under load, requests pile up and health responses become erratic. The first hypothesis is that Ray cannot schedule enough replicas. Inspection of the handler and event-loop lag shows that the replica is monopolized by its own code.

The fix is to use an async client for network waits, move CPU-heavy work to an appropriate thread or process path, or increase replicas after benchmarking. A larger request limit is only useful if the replica can make progress on those requests. If the model itself is synchronous and CPU-bound, a `def` handler or dedicated execution strategy may be clearer. The lesson is that async is cooperative. It does not interrupt a blocking call or manufacture extra CPU cores. The [asyncio guide](https://docs.ray.io/en/latest/serve/advanced-guides/asyncio-best-practices.html) is the right reference for these tradeoffs.

### 11.5. Retries turned a short overload into a long outage

A gateway sends 500 requests per second to a service that can complete 400. The service begins rejecting excess requests quickly, but every client retries each failure three times without jitter. Offered traffic becomes much higher than the original 500 requests per second. A short spike now persists because retries compete with first attempts and newly available replicas. The initial hypothesis is that the service needs a much higher `max_replicas`. Capacity may help, but uncontrolled retries can consume any finite headroom.

The fix is coordinated admission and client behavior. Keep a bounded queue, communicate deliberate shedding with an appropriate status and `Retry-After` policy, cap attempts, add exponential backoff with jitter, and stop retrying after the user's deadline. Distinguish idempotent requests from requests that may have produced a partial streamed response. Test the gateway configuration too: a policy that retries every 5xx may replay intentional shedding. The lesson is that the retry policy is part of the server's effective load curve, not a harmless client detail. Ray's [production guide](https://docs.ray.io/en/latest/serve/production-guide/best-practices.html) explicitly discusses retries, backoff, and load-shedding responses.

### 11.6. A healthy rollout had an unhealthy first request

A new generator image rolls out successfully. Serve reports healthy replicas, and Kubernetes shows running pods. The first few customer requests still take ten seconds, while subsequent requests complete in under a second. The first hypothesis is a transient router problem. The real cause is deferred model initialization: a tokenizer asset is fetched and an inference kernel is compiled on the first real request. The health probe only checks that the process answers a lightweight endpoint.

The fix is to initialize required assets before readiness and send representative warm-up traffic in the deployment workflow. Record model load time and first-request time separately. Reserve enough resources for old and new replicas to overlap during the update. If a full warm-up is too expensive, disclose a bounded cold-start expectation and keep a warm replica floor for interactive traffic. The lesson is that “running” and “ready for the production payload” are different states. A health check is useful only when it exercises the dependency whose failure would make users wait or fail.

## 12. A practical prelaunch checklist

**Rule of thumb: every launch claim needs a corresponding measurement or failure drill.** A Serve config that parses is only the beginning.

1. **State the workload.** Record request arrival distribution, payload sizes, model versions, prompt and output lengths if applicable, and the p95 or p99 latency objective.
2. **Benchmark one replica.** Measure throughput, processing latency, peak RAM or VRAM, startup time, and sensitivity to concurrency. Use the same input distribution as production.
3. **Budget the graph.** Allocate latency and queue budgets to ingress, each deployment, external calls, and response streaming. Confirm that the sum leaves margin for variability.
4. **Bound overload.** Set `max_ongoing_requests` and `max_queued_requests` from measurements. Test the status code and client retry behavior when the queue fills.
5. **Prove placement.** Request the intended CPU/GPU resources and confirm the cluster can place peak replicas. Test what happens when a node or GPU disappears.
6. **Exercise scale-up.** Measure demand detection, new-node provisioning, image pull, model initialization, and first useful request. Compare that total with the burst duration and SLO.
7. **Test batching at low load.** Measure formation wait, actual batch-size distribution, p99, and memory at both quiet and peak traffic.
8. **Version the deployment.** Pin the Ray image and model artifacts. Deploy through a reviewed Serve config and, on Kubernetes, a RayService manifest.
9. **Watch the correct signals.** Graph request rate, errors, queue, processing latency, desired and running replicas, resource utilization, and first-token metrics where relevant.
10. **Practice rollback.** Inject a model-load failure, a slow dependency, and a capacity shortage. Verify that alerts identify the cause and that a known-good revision can return traffic to the SLO.

This checklist is intentionally testable. “Autoscaling is enabled” is not evidence. “The service added four ready replicas within 35 seconds under a measured burst while p99 stayed below 800 ms” is evidence. The numbers in that sentence are examples of what a launch report should contain, not results measured for this article.

### When to reach for Ray Serve

Use it when several model or business-logic stages genuinely need independent resource reservations or replica counts; when Python composition is central to the endpoint; when deployment handles simplify internal calls; or when Ray cluster scheduling and Serve LLM's distributed patterns match a concrete requirement. It is also attractive when a team already operates Ray and can reuse its observability and deployment practices. The [official overview](https://docs.ray.io/en/latest/serve/index.html) presents these as Serve's central strengths.

### When a smaller server is enough

A single model on one host, a strict sub-millisecond routing budget, or a team without a need for Ray's cluster operations may be better served by a direct engine endpoint or a small API server. An offline batch pipeline belongs in a batch system. If the problem is poor kernel efficiency, Serve will not fix the kernel. If the problem is an external database, independent model replicas will not fix the database. Choose the smallest layer that owns the measured bottleneck. For a broader comparison, see [choosing your serving stack](/blog/machine-learning/inference-frameworks/choosing-your-serving-stack).

> A serving system earns its complexity only when it lets us change capacity at the place where the work actually waits.

## Further reading

- [Ray Serve overview and quickstart](https://docs.ray.io/en/latest/serve/index.html)
- [Ray Serve architecture](https://docs.ray.io/en/latest/serve/architecture.html)
- [Production best practices and load shedding](https://docs.ray.io/en/latest/serve/production-guide/best-practices.html)
- [Autoscaling guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html)
- [Dynamic batching guide](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html)
- [Monitoring guide](https://docs.ray.io/en/latest/serve/monitoring.html)
