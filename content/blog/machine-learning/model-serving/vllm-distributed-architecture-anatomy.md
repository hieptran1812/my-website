---
title: "The anatomy of vLLM: How V1 is built as a distributed system"
date: "2026-07-06"
publishDate: "2026-07-06"
description: "Trace a single request through vLLM V1's five process layers, the control plane and the data plane, and the broadcast-then-gather executor, so you can reason about, tune, and debug a multi-GPU LLM server instead of treating it as a black box."
tags:
  [
    "model-serving",
    "inference",
    "vllm",
    "distributed-systems",
    "tensor-parallelism",
    "gpu-inference",
    "llm-serving",
    "multiprocessing",
    "nccl",
    "zmq",
    "ray",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/vllm-distributed-architecture-anatomy-1.webp"
---

The page came in at 03:12: "vLLM pod stuck at 0 tokens/s, GPUs at 100% util, no output." A Llama-3.1-70B deployment on four H100s had been serving fine for a week; a routine config change had flipped `--distributed-executor-backend` and the whole thing wedged. The on-call engineer did the reasonable thing — `kubectl logs` — and found four nearly identical stack traces, one per GPU, all blocked inside an NCCL collective, plus one lonely line in the API server about a socket timeout. Nobody on the team could say which process was waiting on which, because to them vLLM was a single magic command: `vllm serve`. The GPUs showing 100% util were not doing work; they were spinning inside a hung all-reduce, waiting for a fourth rank that had already died. It took two hours to figure that out. It should have taken ten minutes.

That incident is the whole reason for this post. A modern LLM server is not one program. It is a distributed system running on a single box (and sometimes across boxes), with several operating-system processes, two completely different communication channels, and a request that crosses process boundaries three or four times before a single token comes back to the client. If you cannot name those processes and channels, every multi-GPU incident becomes a two-hour archaeology dig. If you can, most of them become a ten-minute read of the right log.

By the end of this article you will be able to draw vLLM V1 from memory: the five layers from the FastAPI route down to one worker process per GPU (Figure 1), the control plane made of ZMQ sockets and shared-memory queues versus the data plane made of NCCL collectives, and the executor pattern that broadcasts one work item to every GPU and gathers the answer from exactly one of them. We will trace a real request across the processes, derive *why* the V1 architecture is built the way it is (the argument comes down to the Python GIL and blocking NCCL calls), and finish with the decision every operator eventually faces: `UniProcExecutor` versus `MultiProcExecutor` versus the Ray backend. Everything here is grounded in the vLLM team's own ["Anatomy of vLLM"](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) write-up and the V1 engine design; where I quantify something the source does not, I flag it as an estimate.

This is a systems post, so keep the serving SLO triangle in view the whole way: **latency, throughput, and cost trade against each other**, and every design choice vLLM makes — multiprocess workers, shared-memory broadcast, single-rank gather — is a bet on that triangle. The payoff of understanding the machinery is that you stop tuning flags by superstition and start tuning them because you know which process the bottleneck lives in.

![Layered diagram of vLLM V1 showing five stacked layers from the FastAPI API server down through AsyncLLM and its client, EngineCore, the executor, and one worker process per GPU](/imgs/blogs/vllm-distributed-architecture-anatomy-1.webp)

## 1. The five layers, top to bottom

Start with the shape of the whole thing. A vLLM V1 server is a stack of five layers, and Figure 1 lays them out in the order a request travels. Each layer has a job, a set of classes that do that job, and — crucially — a process it lives in. Naming all three for each layer is the single most useful thing you can carry away from this post, because a serving bug is almost always "the wrong thing happened in *this* layer, in *this* process," and you cannot say that sentence until you know the layers.

The **API server layer** is the front door. It is a FastAPI application served by Uvicorn (an ASGI server running an asyncio event loop). When a client POSTs to `/v1/completions` or `/v1/chat/completions`, the request lands in one of two handler classes: `OpenAIServingCompletion` or `OpenAIServingChat`. The function `create_completion` validates the payload, tokenizes the prompt, assembles the request metadata — a unique request ID, the sampling parameters, a timestamp — and hands it downward. This layer speaks HTTP to the outside world and speaks to the next layer down over an in-process Python call plus, as we will see, a socket. It does no model math. It is pure I/O orchestration, which is exactly why it lives on an asyncio event loop: it spends its life waiting.

The **AsyncLLM and client layer** is the boundary between the web world and the engine world. `AsyncLLM` is the asynchronous engine handle the API server holds. When `create_completion` calls `AsyncLLM.generate()`, that call does not run a model; it registers the request and returns an async generator that will yield tokens as they arrive. Internally `AsyncLLM` owns a `DPLBAsyncMPClient` — the name unpacks to "**d**ata-**p**arallel, **l**oad-**b**alancing, **async**hronous, **m**ulti**p**rocessing client." That mouthful is the whole design brief of the layer: it can talk to several data-parallel engine replicas, it picks which one gets a request by a load-balancing rule, it does everything asynchronously so the event loop never blocks, and it communicates with the engine *across a process boundary* using ZMQ sockets. This client is where a request stops being an in-process function call and becomes a message sent to another process.

The **EngineCore layer** is where scheduling and the forward pass actually happen. `EngineCore` holds the scheduler, the KV-cache block manager, and a handle to the executor. In a multiprocess deployment it runs inside its own process wrapped by `EngineCoreProc` (and its data-parallel subclass `DPEngineCoreProc`), and that process is started by the function `run_engine_core()`, which builds the object and then enters a busy loop. The heartbeat of that loop is `engine_core.step()`: schedule the next batch, run one forward pass through the executor, post-process the outputs, repeat. Everything about vLLM's throughput — continuous batching, prefix caching, chunked prefill — is a decision made inside `step()`. If you have read the companion [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) deep-dive, the scheduler in that post is the thing that runs here, once per step, inside this process.

The **executor layer** is the fan-out point. `EngineCore` does not talk to GPUs directly; it talks to an `Executor`, and the executor's whole job is to take one piece of work — "run this forward pass on this batch" — and make it happen on however many GPUs the model needs. There are two implementations, and choosing between them is most of the operational story of this post. `UniProcExecutor` is the trivial case: one process, one `Worker`, one GPU; `execute_model()` is a direct method call. `MultiProcExecutor` is the interesting case: it spawns `world_size` daemon worker processes (one per GPU rank), broadcasts each work item to all of them through a shared-memory message queue called `rpc_broadcast_mq`, and gathers the result from a designated rank's `worker_response_mq`. The executor is the layer that turns "one engine" into "N GPUs working in lockstep."

The **worker layer** is the muscle. Each `Worker` owns one GPU, one shard of the model weights, and one slice of the KV cache. In a `MultiProcExecutor` deployment each worker runs inside its own `WorkerProc` process — one OS process per GPU — spawned by `WorkerProc.make_worker_process()`. Rank 0 is special: it is the driver / designated output rank, the one whose result the executor actually reads back; the other ranks compute their shard and participate in the collective communication but do not return a result to the parent. Between the workers runs the second communication channel, the data plane: NCCL collectives (all-reduce, all-gather) that stitch the sharded computation back into a whole. Workers are where the FLOPs happen and where, when tensor parallelism is misconfigured, the whole server silently hangs.

Read Figure 1 top to bottom and you have the itinerary of every request: FastAPI route → `AsyncLLM`/client → `EngineCore` → executor → workers → back up. Five layers, and at least two hard process boundaries in between (API/client process ↔ engine process, engine process ↔ worker processes). The rest of this post is about what happens *at* those boundaries, because that is where all the interesting engineering — and all the interesting failures — live.

Here is the simplest possible entry point, the synchronous `LLM` API, which quietly stands up this entire stack:

```python
from vllm import LLM, SamplingParams

# One call builds the whole stack in Figure 1: an EngineCore, a
# MultiProcExecutor, and four WorkerProc processes (one per GPU).
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,              # -> world_size = 4 workers
    distributed_executor_backend="mp",   # MultiProcExecutor (the default on 1 node)
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain tensor parallelism in two sentences."], params)
print(outputs[0].outputs[0].text)
```

That `tensor_parallel_size=4` is not a hint; it is a decision that spawns four operating-system processes, opens a shared-memory broadcast queue, and initializes a four-way NCCL communicator before your first token is ever generated. The rest of this post explains what that one line set in motion.

## 2. Why V1 is multiprocess: the GIL and the blocking-collective argument

Before tracing a request, we have to answer the question the architecture answers: why does vLLM run one process per GPU at all? Threads would be simpler — one process, one thread per GPU, shared memory for free. vLLM V0 essentially did that, and vLLM V1 deliberately tore it apart into separate processes. The reason is two properties of the Python + CUDA + NCCL stack that fight each other, and Figure 2 contrasts the two designs.

![Before-and-after comparison contrasting V0 as a single GIL-bound Python process holding the engine and all workers against V1 as a multiprocess design with one worker process per GPU](/imgs/blogs/vllm-distributed-architecture-anatomy-2.webp)

The first property is the **Global Interpreter Lock (GIL)**. CPython allows exactly one thread to execute Python bytecode at a time. That is fine for I/O-bound code, because a thread waiting on a socket releases the GIL. It is poison for the CPU-side work of driving multiple GPUs, because launching CUDA kernels, building the next batch's metadata, updating the block tables, and sampling tokens are all Python-heavy operations. In a single-process, multi-threaded design, the thread driving GPU 0 and the thread driving GPU 1 cannot both advance their Python launch code simultaneously — they take turns holding the one GIL. The launch work for N GPUs serializes onto a single interpreter even though the GPUs themselves are independent.

### The mechanics: quantifying the serialized launch

Put numbers on it. A single decode step for a large transformer issues on the order of a few hundred CUDA kernel launches per GPU (attention, the MLP, layernorms, the sampling kernels), and each launch carries some Python-side dispatch cost — call it $c$ microseconds per launch through the framework, realistically in the tens of microseconds once you count PyTorch's dispatcher and vLLM's own bookkeeping. If a decode step's CPU-side launch work per GPU is

$$
L = n_{\text{kernels}} \cdot c \approx 300 \times 30\,\mu s \approx 9\ \text{ms},
$$

then in a GIL-serialized design across $N$ GPUs the CPU critical path per step is roughly $N \cdot L$, because only one thread advances Python at a time. For ${N = 4}$ that is about ${36}$ ms of pure CPU launch overhead per step. If the GPU compute for that same decode step is, say, ${10}$–${15}$ ms, then the CPU side is the bottleneck: the GPUs finish their math and sit idle waiting for the single interpreter to hand them the next batch of kernels. That idle time is the "launch bubble" in the left panel of Figure 2, and it is why a GIL-bound design leaves large GPUs perhaps 30% idle at decode. (These are order-of-magnitude figures to make the mechanism legible, not measured constants; the exact split depends on model size, batch size, and how much has been fused into CUDA graphs.)

Now give each GPU its own process. Each process has its own interpreter and its own GIL. The launch work runs truly in parallel, so the CPU critical path per step drops from $N \cdot L$ back toward $L$ — you have recovered a factor of $N$ on the CPU side. That is the entire throughput argument for the multiprocess design, and it is why the right panel of Figure 2 shows the GPUs near 95% busy: the CPU is no longer the serial resource.

### From step rate to throughput: the Little's Law tie-in

The reason this matters for the SLO triangle is that the step loop's rate is a hard cap on server throughput, and CPU overhead sets that rate. Let $T_{\text{step}}$ be the wall-clock time of one `engine_core.step()` — the maximum of the GPU compute time and the CPU launch time, since the two overlap only partially. If the CPU launch path is $N \cdot L = 36$ ms and the GPU compute is 12 ms, then $T_{\text{step}} \approx 36$ ms and the server runs at ${\approx 28}$ steps/second; the GPUs are idle two-thirds of every step. Get the CPU path down to $L = 9$ ms and $T_{\text{step}} \approx 12$ ms (now GPU-bound), so the server runs at ${\approx 83}$ steps/second — roughly a 3x step-rate improvement purely from removing the serial interpreter, and the GPUs are now the bottleneck, which is exactly where you want them.

Little's Law closes the loop from step rate to concurrency. In a continuous-batching server the number of requests in flight is $L_{\text{req}} = \lambda \cdot W$, where $\lambda$ is arrival rate and $W$ is mean time in system. Each in-flight request consumes one slot in the running batch and its KV-cache footprint; the sustainable $\lambda$ is bounded by how many tokens/second the step loop can emit, which is (batch size) × (step rate). Tripling the step rate at a fixed batch size roughly triples the token throughput, which either lets you serve 3x the arrival rate at the same latency or hold the same rate at a fatter batch for better cost-per-token. This is why the V0→V1 rewrite shows up as a throughput number and not a latency number: it did not make a single request faster; it made the step loop faster, so more requests fit under the same GPU. Every optimization in this post — multiprocess workers, CUDA-graph capture in worker boot, shared-memory queues instead of sockets on the hot path — is aimed at shrinking $T_{\text{step}}$ so Little's Law hands you more concurrency for free.

The second property is that **NCCL collectives are blocking C calls**. When four workers perform an all-reduce, each one enters a NCCL call that does not return until the collective completes. In a single-process threaded design, a thread inside a blocking NCCL call interacts badly with the GIL and with the other GPUs' launch threads — you get contention and stalls precisely when you most need all four GPUs to march in step. In a multiprocess design, each worker process blocks in its own NCCL call independently; there is no shared interpreter for them to contend over. The collective becomes clean: everyone arrives, the reduction happens on the GPUs over NVLink, everyone leaves.

So the V1 design is not multiprocess for fashion. It is multiprocess because (a) the GIL serializes the CPU-side launch work that would otherwise parallelize across GPUs, and (b) blocking NCCL calls need to not contend for a shared interpreter. The cost you pay for solving both is that the layers can no longer talk by sharing Python objects; they must talk over inter-process channels. That cost is exactly the control plane, which is the subject of the next section.

Here is the V0-versus-V1 trade laid out as a table, because the differences compound:

| Property | V0 (single-process, threaded) | V1 (multiprocess) |
|---|---|---|
| Process model | One process; scheduler + all workers as threads | EngineCore process + one `WorkerProc` per GPU |
| CPU launch path (N GPUs) | Serialized on one GIL, ~$N \cdot L$ | Parallel across processes, ~$L$ |
| NCCL collectives | Blocking calls contend with GIL / launch threads | Each process blocks independently |
| Inter-layer comms | Shared Python objects, direct calls | ZMQ sockets + shared-memory queues |
| Scheduler / step overhead | On the request's critical path | Overlapped in a dedicated busy loop |
| Reported throughput | Baseline | Up to ~1.7x higher on some models (vLLM V1 alpha) |
| Failure surface | One process crashes → everything dies together | A worker can hang while others wait — needs cross-process debugging |

The last two rows are the real trade. V1 buys throughput and clean scaling; it pays with a genuinely distributed failure surface. That is the whole reason the anatomy is worth learning: the throughput win is automatic, but the debugging tax is only cheap if you understand the process layout. If you want the operational side of that tax in depth, the sibling post on [debugging vLLM distributed serving](/blog/machine-learning/model-serving/debugging-vllm-distributed-serving) walks through the log patterns of a hung collective.

## 3. Two planes: control over ZMQ, data over NCCL

Once the layers live in different processes, they need wires between them. vLLM runs two entirely separate kinds of wire, and keeping them straight is the key to reading any distributed-serving problem. Figure 3 draws both.

![Graph showing the control plane where small messages travel over ZMQ sockets and the rpc_broadcast_mq shared-memory queue, alongside the data plane where large tensors flow worker-to-worker over NCCL collectives](/imgs/blogs/vllm-distributed-architecture-anatomy-3.webp)

The **control plane** carries small messages: "here is a new request," "here is a batch to run," "here are the tokens you produced." These payloads are bytes to kilobytes — request IDs, sampling parameters, token lists, control commands like `ADD` or `ABORT`. They need low latency and asynchronous fan-out and fan-in across processes on the *same host*. vLLM builds the control plane from two mechanisms. Between the frontend (API server / `AsyncLLM` client) and the engine, it uses **ZMQ DEALER sockets** — an asynchronous message pattern where either side can send without waiting for a reply, which fits the streaming, many-requests-in-flight nature of LLM serving. Inside the executor, from the parent (`MultiProcExecutor`, living in the engine process) to the worker processes, it uses a **shared-memory message queue called `rpc_broadcast_mq`** for broadcasting work, and one **`worker_response_mq` per worker** for sending results back. Shared memory is the right tool there because the parent and workers are on the same machine and the messages are frequent; a shared-memory ring buffer avoids kernel socket overhead on the hottest path.

The **data plane** carries something completely different: the tensors. When four workers run a tensor-parallel layer, each computes a partial result over its shard of the attention heads, and those partials must be summed across GPUs — an all-reduce — before the next layer can proceed. These payloads are large: activations are on the order of (batch × sequence × hidden) elements, megabytes per layer per step, and they must move GPU-to-GPU at the speed of NVLink or InfiniBand, not through the CPU. This is exactly what NCCL is for. The data plane is **NCCL collectives running directly between the workers' GPUs**, and — this is the important part — it never touches ZMQ or the shared-memory queues. The tensors do not flow through the engine process. They flow worker-to-worker, on the hardware fabric, coordinated only by a tiny bit of control-plane signaling.

### The mechanics: why not one channel for both?

You might ask why vLLM does not just use one mechanism for everything. The math answers it. Suppose you tried to move the all-reduce traffic over ZMQ through the engine process. A single decode step of a 70B model with tensor parallelism moves on the order of tens of megabytes of activation data per all-reduce, several times per layer, across 80 layers. Routing that through a CPU process means serializing the tensor to bytes, copying it host-to-device and device-to-host, and pushing it through a socket — you would add hundreds of microseconds to milliseconds of pure copy-and-serialize overhead per collective, and you would saturate the CPU's memory bandwidth doing nothing but shuttling activations. NCCL instead does the reduction on the GPUs over a 900 GB/s NVLink fabric with zero CPU involvement. Using the wrong channel here is not a small inefficiency; it is a category error that would make multi-GPU serving pointless.

The reverse is equally true. You would not use NCCL for control messages. NCCL collectives are synchronous and symmetric — every rank must participate, in the same order, or the whole thing deadlocks. Control traffic is asynchronous and asymmetric: a new request arrives at an unpredictable time, gets routed to one engine, and produces a stream of tokens back. That is a message-bus workload, and ZMQ is a message bus. Trying to express "one new request arrived" as a collective would be absurd. So the two planes are not an accident of history; they are two workloads with opposite requirements — small/async/CPU-routed versus large/sync/GPU-fabric — and vLLM gives each the mechanism that fits.

There is a third detail that makes the control plane inside the engine process work: **threads**. `EngineCoreProc` runs three cooperating threads connected by Python queues and a `threading.Event` for coordination. The **input thread** blocks on the input ZMQ socket; when a message arrives it decodes it and puts a work item on an in-process `input_queue`. The **main thread** is the busy loop: it pulls from `input_queue`, adds the request to the engine, and calls `engine_core.step()` repeatedly, pushing results onto an `output_queue`. The **output thread** blocks on `output_queue` and sends results out over the output socket. This input → main → output pipeline means socket I/O never blocks the step loop and the step loop never blocks socket I/O — a classic decoupling that keeps the GPU fed. A `threading.Event` (a `ready_event`, for example) coordinates startup so the main thread does not begin stepping until the sockets are wired up and the handshake is done.

Why DEALER sockets specifically, and not a plain request/reply pair? A ZMQ REQ/REP socket is strictly lockstep — send one request, wait for exactly one reply, repeat — which is fatal for LLM serving, where the frontend has hundreds of requests in flight and each produces a stream of many token messages back, not a single reply. DEALER (paired with a ROUTER on the other end) is fully asynchronous and multiplexed: either side can fire messages whenever it wants, tagged by request, and the socket fans them in and out without imposing an order. That is precisely the shape of the workload — many concurrent requests, streaming partial outputs, arriving and completing out of order. The startup handshake exists because both ends must agree on the socket addresses and confirm the peer is alive before the first `ADD` is sent; a mismatched address or a peer that died during boot shows up here as a hang in the handshake rather than a dropped request later, which is easier to diagnose. Backpressure lives on this plane too: if the engine's `input_queue` fills faster than the main thread drains it (arrivals outrunning the step loop), the pressure propagates back through the socket buffers to the client, which is the honest signal that you are past the engine's sustainable arrival rate — the moment to scale out replicas rather than let latency silently balloon.

So the control plane, fully spelled out, is: ZMQ DEALER sockets between frontend and engine; a three-thread input/main/output pipeline inside the engine process; `rpc_broadcast_mq` from the executor to the workers; and `worker_response_mq` back from each worker. The data plane is: NCCL collectives directly between worker GPUs. Every message in vLLM travels on exactly one of these, and knowing which is the difference between "the API server can't reach the engine" (control plane, a ZMQ problem) and "the GPUs are hung in an all-reduce" (data plane, an NCCL problem). Figure 4 turns this into a lookup table you can keep next to the runbook.

![Matrix mapping each vLLM layer to its process, its control-plane communication mechanism, its data-plane mechanism, and its key class](/imgs/blogs/vllm-distributed-architecture-anatomy-4.webp)

The matrix in Figure 4 is worth memorizing row by row, because it is the diagnostic key. Notice that only the executor and worker rows have anything in the data-plane column — NCCL — and everything above them is control-plane only. That single fact tells you that any "hang with GPUs busy" is a bottom-two-rows problem, and any "requests never reach the model" is a top-three-rows problem. You have cut the search space in half before reading a single line of the actual log.

Here is the same mapping as a compact reference table, since a table survives copy-paste into a wiki better than an image:

| Layer | Process | Control-plane comms | Data-plane comms | Key class(es) |
|---|---|---|---|---|
| API server | API/asyncio process | HTTP in; in-process call down | none | `OpenAIServingCompletion`, `OpenAIServingChat` |
| AsyncLLM client | API/asyncio process | ZMQ DEALER socket to engine | none | `AsyncLLM`, `DPLBAsyncMPClient` |
| EngineCore | Engine process (busy loop) | input/output ZMQ sockets; 3 threads | none | `EngineCore`, `EngineCoreProc`, `run_engine_core` |
| Executor | Inside engine process | `rpc_broadcast_mq` (shared memory) | coordinates NCCL | `MultiProcExecutor`, `UniProcExecutor` |
| Worker (rank) | One process per GPU | `worker_response_mq` back to parent | NCCL all-reduce / all-gather | `WorkerProc`, `Worker` |

## 4. Tracing one request across the processes

Now walk a single completion request through the whole machine. Figure 5 is the map; follow it node by node as the prose narrates each hop, and note every place the request crosses a process boundary.

![Graph tracing one request from create_completion through AsyncLLM.generate and the load-balancing pick, into the engine's input socket and step loop, out through the executor to the workers, and back via the gather to the JSON response](/imgs/blogs/vllm-distributed-architecture-anatomy-5.webp)

A client POSTs to `/v1/completions`. Uvicorn's event loop dispatches it to `OpenAIServingCompletion`, and `create_completion` runs: it validates the JSON, tokenizes the prompt asynchronously (tokenization is CPU work, so it is done off the event loop's critical path), and builds the request metadata — a fresh request ID, the `SamplingParams`, a timestamp. This is all still in the API process, on the asyncio loop.

`create_completion` calls `AsyncLLM.generate()`. This does not run the model. It calls `DPLBAsyncMPClient.add_request_async()`, which is where the load balancer picks an engine. If you are running data-parallel replicas — several complete engine copies behind one front end — the client must choose which one gets this request. The rule is deliberately simple and worth committing to memory:

```
score = len(waiting) * 4 + len(running)
```

The engine with the **lowest** score wins (`get_core_engine_for_request()` selects it). The weighting says a request sitting in the waiting queue counts four times as much as a request already running, because a waiting request represents unstarted work — latency the client has not yet begun to pay down — whereas a running request is already being amortized across the continuous batch. Penalizing waiting work four-to-one steers new requests toward engines that can start them immediately, which is what protects time-to-first-token (TTFT) under load. Once an engine is chosen, `add_request_async` sends an `ADD` message to that engine's **input socket** over ZMQ. The request has now left the API process.

Inside the chosen engine process, the three threads take over. The **input thread** was blocked on the input socket; the `ADD` message unblocks it. It decodes the message and places the work item on the in-process `input_queue`. The **main thread** — the busy loop — pulls the item off `input_queue`, registers the request with `EngineCore` (the scheduler now knows about it and will include it in a future batch), and continues calling `engine_core.step()`. On some step, the scheduler decides this request's prompt should be prefilled, or its next token decoded, and includes it in the batch it hands to the executor.

Here is the fan-out. `engine_core.step()` calls the executor's `execute_model()`. On a `MultiProcExecutor`, that enqueues the work item onto `rpc_broadcast_mq`, the shared-memory broadcast queue. All the worker processes are blocked reading that queue; the enqueue wakes all of them at once. Each worker runs its shard of the forward pass on its GPU, and where the model requires it, the workers all-reduce over NCCL — that is the data plane doing its job while the control plane waits. When the forward pass finishes, only the **designated output rank** (rank 0) enqueues its result onto its `worker_response_mq`; the executor parent, back in the engine process, was blocked on exactly that queue's `dequeue()` and now wakes up with the batch's output tokens. The other ranks computed identical logits (after the all-reduce, every rank holds the full result) but do not bother sending them back — reading one is enough, and reading four would waste bandwidth.

The main thread post-processes the step's output — samples tokens, updates the block tables, checks stop conditions — and pushes the per-request results onto `output_queue`. The **output thread** picks them up and sends them back over the output socket to the API process. There, `AsyncLLM`'s output-handling task (the reference calls out `process_outputs_socket()` reading the socket and `output_handler()` propagating tokens) resolves the async generator that `create_completion` has been awaiting, and each new token streams out to the client. When the sequence finishes, FastAPI returns its final `JSONResponse` (or closes the SSE stream). The request is done, having crossed from the API process to the engine process to the worker processes and all the way back.

Count the boundaries: API process → (ZMQ) → engine process → (shared-memory queue) → worker processes → (shared-memory queue) → engine process → (ZMQ) → API process. Three transport hops each way, two of them shared-memory and one ZMQ. Every one of those hops is a place a message can be dropped, a socket can time out, or a queue can back up — which is precisely why knowing the path matters when something goes wrong.

The streaming detail is what makes this feel different from a classic RPC. `AsyncLLM.generate()` does not return one answer; it returns an async generator that yields as tokens arrive, and the engine emits output on *every* step the request participates in, not just at the end. So the loop is not "request in, response out" but "request in, then a stream of incremental outputs out, one per decode step, until the stop condition fires." On the API side, `process_outputs_socket()` reads each incremental batch off the output socket and `output_handler()` routes each request's new tokens to the right waiting generator, which is how one engine feeds hundreds of concurrent SSE streams without any of them blocking the others. Abort is the mirror image and easy to overlook: if the client disconnects mid-generation (closes the browser tab, hits a timeout), FastAPI cancels the async generator, `AsyncLLM` sends an `ABORT` control message down the same ZMQ socket, and the engine removes the request from the running batch so its KV-cache blocks are freed immediately. Without that path, a flood of clients that connect, prompt, and disconnect would leak KV cache and slowly strangle the server — so the abort message is not a nicety, it is the backpressure valve that keeps a misbehaving client from consuming capacity it abandoned. Every one of these messages is control-plane traffic; none of it touches NCCL.

#### Worked example: the load balancer picks an engine

Concrete numbers make the score rule click. Suppose a data-parallel deployment has two engine replicas behind one `DPLBAsyncMPClient`, and a burst of traffic arrives. At the instant a new request shows up:

- **Engine A** has 2 requests waiting and 30 running: `score = 2 * 4 + 30 = 38`.
- **Engine B** has 0 requests waiting and 44 running: `score = 0 * 4 + 44 = 44`.

Engine B is running more total requests (44 versus 32), so a naive "least busy by total count" rule would still pick B — wrong. The score picks **A** (38 < 44), because A can start the new request *now* (nothing is queued ahead of it once it schedules), whereas B, despite having zero waiting, is carrying a heavier running batch that will lengthen each decode step. Now flip it: if Engine A had 10 waiting and 30 running, its score would be ${10 \times 4 + 30 = 70}$, and B at 44 would win decisively — A is backlogged, and shoving another request at it would only grow the queue. The four-to-one weight is what makes the rule track *TTFT risk* rather than raw load. For a deployment doing 1,000 requests/second across replicas, that difference is the difference between a p99 TTFT of 200 ms and one of 2 seconds. This is the same least-outstanding-work instinct behind good HTTP load balancers, specialized for the fact that a queued LLM request is far more expensive to a newcomer than an in-flight one.

## 5. The executor: broadcast to all, gather from one

The executor deserves its own section because it is the single most important idea for reasoning about multi-GPU vLLM, and it is where the "distributed system on one box" character is sharpest. The `Executor` presents one interface to `EngineCore` — `execute_model()`, `collective_rpc()`, a handful of others — and hides how many GPUs are behind it. Swap `UniProcExecutor` for `MultiProcExecutor` and `EngineCore` does not change a line; that uniform interface is the whole point.

`UniProcExecutor` is the degenerate case: one process, one `Worker`, one GPU. `execute_model()` is a direct Python method call on the worker. No IPC, no serialization, no NCCL. If your model fits on one GPU, this is what you want, and it is what you get by default at `tensor_parallel_size=1`.

`MultiProcExecutor` is the real machine. At construction it spawns `world_size` daemon processes — one per GPU rank — via `WorkerProc.make_worker_process()`. Each child runs `WorkerProc.worker_main()`, which instantiates a `Worker`, binds it to its GPU, and joins the NCCL communicator. The parent then drives all of them through the two shared-memory queues: it broadcasts every work item onto `rpc_broadcast_mq`, which all children consume, and it reads results from the designated rank's `worker_response_mq`. Figure 6 shows a single TP=4 forward pass moving through this machine as a grid: broadcast at the top, sharded compute, the NCCL all-reduce across ranks, and the single-rank gather at the bottom.

![Grid showing a TP=4 forward pass: all four ranks dequeue from rpc_broadcast_mq, each computes its QKV head shard, all four all-reduce over NCCL, and only rank 0 enqueues the response the parent reads](/imgs/blogs/vllm-distributed-architecture-anatomy-6.webp)

The pattern is **broadcast-then-gather**, and its asymmetry is the subtle part. Work fans out to *all* ranks: every GPU must run its shard, because tensor parallelism splits each matrix multiply across GPUs and a layer is not complete until every shard has computed and the partials are all-reduced. But the result is gathered from *one* rank. After the all-reduce, every rank holds the identical final logits — that is what all-reduce means, every participant ends with the summed result — so the parent only needs to read one. Reading all four would move four identical copies of the output through the response queues for no reason. Hence: broadcast to `world_size`, gather from rank 0.

Here is the executor loop expressed in code that mirrors the real method names, so you can map it onto the source:

```python
# Sketch of MultiProcExecutor, grounded in vLLM's real method names.
# This is the parent side, living inside the EngineCore process.

class MultiProcExecutor:
    def __init__(self, vllm_config):
        self.world_size = vllm_config.parallel_config.world_size
        # Shared-memory queues: one broadcast queue out, one response queue
        # per worker back. Created before the workers so they can attach.
        self.rpc_broadcast_mq = MessageQueue(n_readers=self.world_size)
        self.workers = []
        for rank in range(self.world_size):
            # Spawn one daemon process per GPU rank. Each child runs
            # WorkerProc.worker_main(), instantiates a Worker, binds a GPU,
            # and joins the NCCL process group.
            proc, response_mq = WorkerProc.make_worker_process(
                vllm_config, rank, self.rpc_broadcast_mq)
            self.workers.append((proc, response_mq))
        self.output_rank = 0  # designated rank whose result we read back

    def execute_model(self, scheduler_output):
        # 1) BROADCAST: enqueue once; all world_size children consume it.
        self.rpc_broadcast_mq.enqueue(("execute_model", scheduler_output))
        # 2) GATHER: block only on the designated output rank's response.
        #    Other ranks compute + all-reduce over NCCL but do not reply.
        _, response_mq = self.workers[self.output_rank]
        model_output = response_mq.dequeue()
        return model_output

    def collective_rpc(self, method, args=(), kwargs=None):
        # Generic "run this method on every worker" primitive. Used for
        # load_model, determine_num_available_blocks, etc. Broadcast to all,
        # collect from all (these results are small and rank-specific).
        self.rpc_broadcast_mq.enqueue((method, args, kwargs or {}))
        return [mq.dequeue() for _, mq in self.workers]
```

Two methods, two patterns. `execute_model()` broadcasts and gathers from one, because the forward-pass output is identical across ranks after the all-reduce. `collective_rpc()` broadcasts and gathers from *all*, because it is used for setup calls where each rank returns something rank-specific and small — for example `determine_num_available_blocks`, where each GPU reports how much KV-cache memory it has free, and the executor takes the minimum so every rank sizes its block pool identically. Knowing which gather pattern a method uses tells you immediately whether a hang is "one rank never replied" (an `execute_model` gather on rank 0) or "some rank in the middle never replied" (a `collective_rpc` gather on all ranks).

The child side is the mirror image — a loop reading the broadcast queue and dispatching:

```python
# Sketch of the worker child loop (WorkerProc.worker_main), one per GPU.
def worker_main(vllm_config, rank, rpc_broadcast_mq, response_mq):
    worker = Worker(vllm_config, rank)      # binds GPU `rank`
    worker.init_device()                    # CUDA device + distributed init
    worker.load_model()                     # weights for this shard
    worker.initialize_cache()               # KV block pool, CUDA graphs

    while True:
        method, *payload = rpc_broadcast_mq.dequeue()   # blocks until work
        result = getattr(worker, method)(*payload)      # run the shard
        if method == "execute_model" and rank != OUTPUT_RANK:
            continue                                     # non-output ranks stay quiet
        response_mq.enqueue(result)                      # reply to the parent
```

Notice the `rank != OUTPUT_RANK: continue`: that one line is the broadcast-then-gather asymmetry made literal. Every rank runs `getattr(worker, method)(...)` — every GPU does the work — but only the output rank puts anything back on its response queue for an `execute_model` call.

Pipeline parallelism bends this pattern without breaking it. Where tensor parallelism splits every layer *across* ranks (so all ranks run every layer on a shard, and the output is identical after all-reduce), pipeline parallelism splits the layers *into stages* along the depth of the model: rank group 0 holds layers 0–39, rank group 1 holds layers 40–79. A forward pass now flows stage to stage — stage 0 computes its half and passes the activations to stage 1 over a point-to-point send, not an all-reduce — and the final logits emerge only from the *last* stage. So under PP the designated output rank is the last pipeline stage, and the executor keeps several microbatches in flight at once to fill the pipeline bubble (while stage 1 works on microbatch 1, stage 0 starts microbatch 2). The executor interface does not change — it is still broadcast a work item, gather from a designated rank — but the "designated rank" moves to the pipeline tail and the data-plane traffic between stages is smaller (a single activation tensor at the layer boundary) than the per-layer all-reduces of TP. That smaller inter-stage payload is exactly why PP, not TP, is the strategy that crosses the slow node boundary; the [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) post works the trade in full.

You can select which executor runs from the command line, and the flag is one of the most consequential you will set:

```bash
# Single node, 4 GPUs -> MultiProcExecutor over shared-memory queues (the default).
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mp

# Same model spanning two 4-GPU nodes -> Ray backend (TP within node, PP across).
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray

# Force the V1 engine and single-process mode explicitly (env vars).
VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    vllm serve mistralai/Mistral-7B-Instruct-v0.3
```

And the asynchronous, production-facing entry point — the one the OpenAI-compatible server actually uses — is the async engine. In V1 the internal class is `AsyncLLM`; the stable public surface remains `AsyncLLMEngine`, which wraps the same machinery:

```python
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# AsyncLLMEngine (public) wraps AsyncLLM (V1 internal). It owns the
# DPLBAsyncMPClient and the ZMQ sockets to the EngineCore process.
engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model="meta-llama/Llama-3.1-70B-Instruct",
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.90,
    )
)

async def stream_completion(prompt: str, request_id: str):
    params = SamplingParams(temperature=0.7, max_tokens=256)
    # .generate() returns an async generator; each item carries the tokens
    # produced so far. This mirrors what create_completion awaits.
    async for output in engine.generate(prompt, params, request_id):
        yield output.outputs[0].text

async def main():
    async for chunk in stream_completion("Describe NCCL all-reduce.", "req-1"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

If you want the full tour of `EngineArgs`, prefix caching, and quantization on top of this skeleton, the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) post is the companion; this post is about the distributed skeleton those features hang on.

#### Worked example: a TP=4 forward pass and its overhead

Trace one decode step of Llama-3.1-70B on four H100s under `MultiProcExecutor`, and account for where the time goes. The scheduler hands `execute_model()` a batch. The parent enqueues one work item on `rpc_broadcast_mq`; a shared-memory ring-buffer enqueue is sub-microsecond to a few microseconds. All four workers dequeue and begin their shard. Each layer's attention and MLP produce a partial that must be all-reduced: 80 layers, and typically two all-reduces per layer (one after attention, one after the MLP), so ~160 collectives per token. On an NVLink-connected H100 node, a single all-reduce of a decode-step activation (batch × hidden, a few hundred KB to low MB) completes in roughly tens of microseconds; call it ~30 µs each, so ~160 × 30 µs ≈ 4.8 ms of NCCL time per token in the data plane. The GPU compute for the step might be ~8–10 ms. The control-plane overhead — one broadcast enqueue, one gather dequeue — is a handful of microseconds, utterly dwarfed by the compute and the collectives.

That ratio is the whole justification for the two-plane split. The data plane (NCCL) is ~5 ms of real, unavoidable communication cost that has to run on the GPU fabric; the control plane (shared-memory queues) is microseconds. If the control messages and the tensor traffic shared one channel, that 5 ms of collective work would have to squeeze through the same microsecond-scale path as the little control messages, serialized and CPU-copied — you would turn a 5 ms fabric operation into tens of milliseconds of CPU thrash. Keeping the planes separate means the expensive part runs on the hardware built for it, and the cheap part stays cheap. The measured lesson: at TP=4, the NCCL data plane, not the vLLM control plane, is your communication budget. If you are latency-bound, that is where to look — faster interconnect (NVLink over PCIe), or fewer collectives (a smaller TP degree if the model fits), not anything about ZMQ or the queues.

## 6. Data parallelism: replicate the engine, load-balance the replicas

Tensor parallelism shards *one* model across GPUs; data parallelism runs *several complete copies* of the engine and spreads requests across them. These are orthogonal, and vLLM composes them: you can run four data-parallel replicas, each of which is itself a TP=2 group, for eight GPUs total. Understanding where each lives in the anatomy is the last piece needed to read a real deployment, because the load-balancing score from Section 4 only makes sense once you know it is choosing among data-parallel replicas.

The `DP` in `DPLBAsyncMPClient` is the point: the client is *data-parallel-aware*. Each replica is a full `EngineCore`, running in its own process — the data-parallel variant is `DPEngineCoreProc` rather than the plain `EngineCoreProc` — with its own scheduler, its own KV cache, and (if it is also tensor-parallel) its own set of worker processes underneath. The client holds a ZMQ socket to each replica's input socket and, when a request arrives, computes `score = len(waiting) * 4 + len(running)` for every replica and sends the request to the lowest. There is no shared batch across replicas; each one continuously batches its own traffic independently. That independence is the strength (no cross-replica coordination on the hot path, so replicas scale linearly) and the reason the load-balancing rule has to be good (a bad routing decision cannot be fixed later — once a request is committed to a backlogged replica, it waits behind that replica's queue).

For the score to be accurate, the client needs fresh `len(waiting)` and `len(running)` numbers from each replica, and those live in the replica's process, not the client's. That is what the `run_engine_stats_update_task()` async task is for: it periodically exchanges queue-depth statistics with a **DP coordinator** so the client's view of each replica's load stays current. Stale stats are a real failure mode — if the client thinks a replica is idle when it is actually backlogged, it will pile requests onto it and spike that replica's TTFT while others sit underused. The coordinator is the small piece of shared state that keeps the otherwise-independent replicas routable.

Contrast the two parallelism strategies directly, because operators conflate them constantly:

| | Tensor parallelism (TP) | Data parallelism (DP) |
|---|---|---|
| What it splits | One model's weights across GPUs | Nothing — full model replicated |
| GPUs per model copy | `tensor_parallel_size` | 1 (or 1 TP group) per replica |
| Communication | NCCL all-reduce every layer (data plane) | none between replicas; ZMQ + stats to client |
| Solves | Model too big for one GPU; decode bandwidth | Throughput / QPS ceiling; horizontal scale |
| Executor | `MultiProcExecutor` (or Ray) | `DPLBAsyncMPClient` load-balances `DPEngineCoreProc` replicas |
| Cost | Collective overhead per token | More total GPU memory (weights replicated N times) |

The decision rule falls out of the table. If the model does not fit on one GPU, you *must* shard — that is TP (or PP across nodes), and it is not optional. If the model fits but you cannot serve the QPS, you replicate — that is DP, and it scales throughput almost linearly because there is no cross-replica traffic. Most large production deployments use both: TP to make the model fit and saturate memory bandwidth, DP to multiply the QPS. And the whole reason the humble `score = len(waiting) * 4 + len(running)` matters is that it is the only coordination point in the DP layer — get it right and replicas scale cleanly; get it wrong and you have expensive GPUs sitting idle behind a bad routing decision.

## 7. Inside a worker: the three-phase boot

A worker process is not ready the moment it spawns. Every `WorkerProc` walks the same startup sequence before it can join the busy loop, and understanding it explains a whole class of "the server takes four minutes to become ready" and "it OOMs during startup, not under load" incidents. Figure 7 lays the sequence out as a timeline.

![Timeline of a worker's startup: init device and set DP/TP/PP/EP, distributed init of the NCCL process group, load the model shard, initialize the KV cache via a profiling pass, capture CUDA graphs, and enter the busy loop](/imgs/blogs/vllm-distributed-architecture-anatomy-7.webp)

**Phase one, init device.** The worker assigns itself a CUDA device (its rank maps to a physical GPU), verifies it can see the VRAM it expects, and sets the distributed configuration — the tensor-parallel (TP), pipeline-parallel (PP), data-parallel (DP), and expert-parallel (EP) sizes that determine how this rank fits into the whole. It instantiates the `model_runner` (the object that will actually drive forward passes) and an `InputBatch` (the reusable structure that holds the current batch's token IDs, positions, and block tables). Part of this phase is joining the NCCL process group: all `world_size` ranks perform a rendezvous so the collective communicator exists before any tensor ever needs reducing. This rendezvous is a common hang point — if one rank cannot reach the others (a firewall, a wrong `MASTER_ADDR`, a dead peer), the process group never forms and every rank blocks here, which is exactly the failure from this post's opening incident.

**Phase two, load model.** The worker instantiates the model architecture and loads the weights for *its shard* — with TP=4, each rank loads roughly a quarter of the parameters, not the whole model. It calls `model.eval()` to switch off training-time behavior (dropout, batchnorm updates), and optionally compiles the model with `torch.compile()` to fuse operations and cut per-op dispatch overhead. Sharded loading is why a 70B model that would never fit on one 80 GB GPU loads fine across four: no single process ever holds all 140 GB of BF16 weights; each holds ~35 GB.

**Phase three, initialize KV cache.** This is the clever one. The worker cannot know how many KV-cache blocks it can afford until it knows how much memory the model and its activations consume, so it runs a **profiling forward pass**: a synthetic maximum-size batch through the model to measure peak memory. Whatever VRAM is left over, after weights and peak activations, becomes the KV-cache block pool. The worker allocates and reshapes those tensors, binds them to the attention layers, prepares the attention metadata, and — for the batch sizes it expects to see — optionally captures **CUDA graphs**: pre-recorded sequences of GPU operations that replay with near-zero launch overhead, which is one more front in the war against the Python launch cost from Section 2. Only after all three phases does the worker enter `run_engine_core()`'s busy loop and start dequeuing real work.

The reason `gpu_memory_utilization` is such a load-bearing flag falls straight out of phase three. That number is the fraction of VRAM vLLM is allowed to use total. Set it to 0.90 and the profiling pass reserves 90% for weights + activations + KV cache; the remaining 10% is headroom for fragmentation and CUDA context. Set it too high and the profiling pass itself can OOM, or you leave no room for a burst of long sequences and the server OOMs under load. Set it too low and you waste KV-cache capacity, capping your batch size and throughput. The flag is a direct knob on the phase-three block-pool sizing, and knowing that turns "0.90 felt safe" into "0.90 leaves ~8 GB of headroom on an 80 GB H100, which covers our worst-case activation spike."

Here is the boot sequence expressed against the real `Worker` and `collective_rpc` calls, which is exactly how the executor drives it during startup:

```python
# How the executor boots every worker in lockstep, via collective_rpc.
# Each call broadcasts to all ranks and gathers small per-rank results.

executor.collective_rpc("init_device")     # phase 1: CUDA device + NCCL group

executor.collective_rpc("load_model")       # phase 2: this rank's weight shard

# phase 3: each rank profiles and reports how many KV blocks it can hold;
# take the MIN so every rank uses an identical block count.
num_blocks_per_rank = executor.collective_rpc("determine_num_available_blocks")
num_blocks = min(num_blocks_per_rank)
executor.collective_rpc("initialize_cache", args=(num_blocks,))

# Only now is the engine ready to accept requests into engine_core.step().
```

The `min()` over per-rank block counts is a small but important invariant: tensor-parallel ranks must have identical KV-cache geometry, so if one GPU happens to have slightly less free memory (a larger CUDA context, another process sharing the card), every rank is capped to that smallest value. It is a reason an otherwise-symmetric deployment can quietly run at lower capacity than the spec sheet suggests — one crowded GPU drags the whole group down to its block count.

#### Worked example: sizing the KV-cache block pool

Put real numbers on phase three for Llama-3.1-70B at TP=4 on 80 GB H100s. The per-token KV-cache footprint of the full model is $2 \times \text{layers} \times \text{kv-heads} \times \text{head-dim} \times \text{bytes} = 2 \times 80 \times 8 \times 128 \times 2 = 320$ KiB (the leading 2 is for K and V; the model uses grouped-query attention with 8 KV heads and head dimension 128 in BF16). With TP=4, the 8 KV heads split 2-per-rank, so each rank holds ${\approx 80}$ KiB of KV per token. vLLM allocates KV in fixed blocks of 16 tokens, so one block costs ${16 \times 80}$ KiB ${= 1.25}$ MiB per rank.

Now the budget. At `gpu_memory_utilization=0.90`, vLLM may use ${\approx 72}$ GB of the 80 GB card. The weight shard is ${140\,\text{GB} / 4 = 35}$ GB per rank, and the profiling pass measures peak activations at, say, ${\approx 4}$ GB. What is left for KV is ${72 - 35 - 4 \approx 33}$ GB per rank. Dividing by the 1.25 MiB block cost gives ${\approx 27{,}000}$ KV blocks per rank, which at 16 tokens each is ${\approx 430{,}000}$ tokens of total KV capacity. That single number sets your concurrency ceiling: you can hold ${\approx 430}$ concurrent sequences of 1,000 tokens, or ${\approx 100}$ of 4,000 tokens, before the block manager starts preempting. Bump `gpu_memory_utilization` from 0.90 to 0.95 and you recover another ${\approx 4}$ GB per rank — roughly 3,000 more blocks, ${\approx 48{,}000}$ more tokens of headroom — at the cost of a thinner activation-spike margin. This is the arithmetic behind every "we hit OOM at peak" and "why is my max batch smaller than I expected" ticket: the block pool is a fixed pie carved out after weights and activations, and both the utilization flag and the TP degree change how big that pie is.

## 8. Putting the anatomy to work: reading the failure modes

The payoff of all this structure is diagnostic speed. Because every symptom belongs to a specific layer, process, and plane, the anatomy is a decision procedure for triage. Walk the common failures against Figure 4's mapping:

A **request that never gets a response, GPUs idle**: the request is stuck above the workers. Check the control plane. Either `create_completion` never reached `AsyncLLM.generate` (an API-layer validation or tokenization error), or the ZMQ socket between the client and the engine dropped (`DPLBAsyncMPClient` cannot reach the input socket), or the engine's input thread is wedged. GPUs idle means the data plane was never engaged; you are debugging ZMQ and the three engine threads, not NCCL.

A **hang with all GPUs at 100% util**: the workers are inside a collective that will not complete — the opening incident. This is a data-plane, bottom-two-rows problem. One rank has entered an all-reduce and is waiting for a peer that died, mis-initialized, or is running a different batch shape. The NCCL process group formed but a participant is missing or desynced. You look at all `world_size` worker logs together, find the one that is not at the same collective as the others, and that is your culprit. Utilization at 100% is the tell that it is NCCL, not the control plane: a spinning collective pegs the GPU.

A **startup that hangs before serving any request**: phase one of the worker boot, the NCCL rendezvous. Every rank is blocked forming the process group. Check `MASTER_ADDR`/`MASTER_PORT`, network reachability between ranks, and whether one worker process crashed during spawn so the group can never reach `world_size`. This is why the `WorkerProc` spawn logs matter: a rank that dies in `make_worker_process` leaves the survivors waiting forever.

An **OOM during startup, not under load**: phase three, the profiling pass, with `gpu_memory_utilization` set too high — or an imbalance where one rank has less free VRAM and the profiling batch does not fit. Lower the utilization, or find the process squatting on the crowded GPU.

Notice the pattern: in every case, naming the layer and plane cuts the search space before you read a single detailed log line. That is the entire practical value of learning the anatomy, and it is why the sibling operational posts — [running vLLM distributed in production](/blog/machine-learning/model-serving/running-vllm-distributed-in-production) and [debugging vLLM distributed serving](/blog/machine-learning/model-serving/debugging-vllm-distributed-serving) — assume this map and build runbooks on top of it.

## 9. Case studies and benchmarks

Ground the design in outcomes. These are drawn from the vLLM team's own writing and public benchmarks; where a number is approximate or version-dependent, I say so.

**vLLM V1's multiprocess redesign (vLLM V1 alpha, 2025).** The V1 rewrite moved the engine off the GIL-bound single-process model of V0 and onto the multiprocess architecture in this post — a dedicated `EngineCore` busy loop, one `WorkerProc` per GPU, ZMQ and shared-memory control planes. The vLLM team reported up to roughly **1.7x higher throughput** on some models relative to V0, attributed largely to slashing per-step CPU overhead so the GPU is no longer starved by the interpreter. The exact multiplier depends heavily on model size and batch (small models with cheap forward passes are more CPU-bound, so they benefit more; a giant model whose forward pass dominates sees a smaller relative gain). The mechanism is exactly the Section 2 argument: recover the factor-of-N you lose to serializing launch work on one GIL.

**The "Anatomy of vLLM" walkthrough (vLLM blog, September 2025).** The reference this post is built on traces the same request path — `create_completion` → `AsyncLLM.generate` → `add_request_async` → the load-balancing score → the engine's input socket and three threads → `engine_core.step()` → the executor's broadcast/gather — and documents the exact class names (`DPLBAsyncMPClient`, `EngineCoreProc`, `MultiProcExecutor`, `WorkerProc`) and the two queues (`rpc_broadcast_mq`, `worker_response_mq`). Its central point is that the same engine interface (`execute_model`) hides whether one GPU or many are behind it: `UniProcExecutor` calls the worker directly; `MultiProcExecutor` calls it through shared-memory queues; `EngineCore` cannot tell the difference. That uniform executor interface is what makes vLLM's scaling story clean.

**Tensor parallelism cost at the collective level (Megatron-LM, Shoeybi et al., 2019; and vLLM TP in practice).** Tensor parallelism as vLLM implements it descends directly from Megatron-LM's sharding of the attention and MLP matmuls, which introduces two all-reduces per transformer block. The worked example in Section 5 quantifies why that matters at serving time: at TP=4 on an NVLink node, the all-reduces are the dominant communication cost per token (single-digit milliseconds), and they scale with the interconnect. The public lesson, consistent across vLLM and TGI benchmarks, is that TP pays off most when the model does not fit on one GPU or when a single GPU's memory bandwidth caps decode throughput — and that going to a higher TP degree than you need *adds* collective overhead without adding capacity. The deep treatment of when TP, PP, and EP each earn their keep lives in the [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) post.

**Multi-node serving of 100B+ models (vLLM distributed + Ray).** For models too large for one node, vLLM composes tensor parallelism *within* a node (fast NVLink all-reduces) with pipeline parallelism *across* nodes (slower inter-node links carry only layer-boundary activations, far less traffic than an all-reduce), and switches the executor to the Ray backend so ranks can live on different machines. The published pattern — TP inside the box, PP between boxes — is a direct consequence of the two-plane thinking in this post: put the heavy, frequent collectives on the fastest fabric, and cross the slow node boundary only where the traffic is small. The full multi-node playbook is in [multi-node LLM serving for 100B+](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus).

**Expert parallelism for MoE models (vLLM EP support).** Mixture-of-experts models add a fourth parallelism axis, expert parallelism (EP), which is why `init_device` sets DP/TP/PP/**EP** together in worker boot. An MoE layer routes each token to a small subset of experts, and EP places different experts on different GPUs, so the forward pass now includes an all-to-all (dispatch tokens to their experts, then combine the results back) on top of any TP all-reduces. In the anatomy this is still data-plane traffic — NCCL collectives between workers — but the collective is an all-to-all rather than an all-reduce, and it is more sensitive to load imbalance because a hot expert can bottleneck the whole step. The two-plane framing holds exactly: the routing decision (which token goes to which expert) is small control-plane-shaped metadata computed on-GPU, while the token activations themselves move over the NCCL data plane. The practical lesson the vLLM and DeepSeek-style deployments report is that EP is essential for serving very large sparse models economically, but it makes the data plane's health — interconnect bandwidth and expert load balance — even more decisive than it is for dense TP. The specifics of routing and memory-efficient MoE serving are their own topic; the point here is that the same executor and two-plane structure absorb a new parallelism axis without a redesign.

The consistent thread across all five: the wins come from matching each kind of traffic to the right channel and from getting the CPU out of the critical path. That is what the anatomy encodes.

## 10. When to use this (and when not to)

The concrete decision the anatomy forces is which executor to run, and Figure 8 lays out the three-way choice. It is not really a preference; it follows from your GPU topology.

![Matrix comparing UniProcExecutor, MultiProcExecutor with the mp backend, and the Ray backend across process model, when to use, communication backend, and scale limit](/imgs/blogs/vllm-distributed-architecture-anatomy-8.webp)

**Use `UniProcExecutor` (implicitly, `tensor_parallel_size=1`) when the model fits on one GPU.** A 7B or 8B model in BF16 is ~16 GB; it fits comfortably on a 24 GB or 40 GB card with room for KV cache. There is zero benefit to multiprocess machinery here and real cost — every all-reduce you add is pure overhead, every extra process is another thing to crash. The single-process executor's direct method call is as fast as it gets. Reach for multi-GPU only when you must, not because it sounds more serious. If you find yourself running TP=2 on a model that fits on one card "for headroom," you are paying collective overhead to solve a problem you do not have; buy KV-cache headroom with `gpu_memory_utilization` or a bigger card instead.

**Use `MultiProcExecutor` with `--distributed-executor-backend mp` when the model needs several GPUs but they are all in one node.** This is the default and the sweet spot for the 70B-class models most teams actually serve. The shared-memory queues are the fastest inter-process control channel available on a single host, and NVLink gives the data plane the bandwidth it needs. Up to a full node — typically 8 GPUs — `mp` is the right answer. It has no external dependencies: no Ray cluster to stand up, no extra failure modes beyond the processes vLLM spawns itself. If your model fits in one node's aggregate VRAM, do not reach past `mp`.

**Use the Ray backend with `--distributed-executor-backend ray` when the model must span multiple nodes.** Once you need more GPUs than one machine has — the 405B and 671B class, or very high TP×PP degrees — the workers have to live on different hosts, and shared-memory queues do not cross machine boundaries. Ray provides the cross-node process management and RPC that `mp` cannot. The cost is real: you now operate a Ray cluster, with its own scheduler, its own failure modes, and its own observability surface, on top of vLLM. Do not pay that cost for a model that fits in one node. The rule of thumb: single node → `mp`; multi node → `ray`; and never multi-node until one node genuinely cannot hold the model.

There is also a "when not to learn this at all" answer, in the spirit of the series. If you are serving a single small model at low QPS behind one GPU, you do not need any of this — `vllm serve` with defaults gives you `UniProcExecutor` and you can stop reading. The anatomy earns its keep the moment you go multi-GPU, because that is the moment the failure surface becomes distributed and the two-hour incidents begin. Learn it before your first multi-GPU deployment, not during your first multi-GPU incident.

One more honest caveat on cost: the multiprocess design is not free even when it is right. You pay startup latency (the three-phase boot per worker, plus the NCCL rendezvous — tens of seconds to minutes for large shards), and you pay a slightly higher operational burden (more processes, more logs, cross-process debugging). The throughput and scaling wins are worth it above one GPU. Below one GPU they are not, and pretending otherwise just adds latency and failure modes to a workload that did not ask for them. That is the SLO triangle again: multiprocess trades a little cold-start latency and operational cost for a lot of steady-state throughput, and you should only take that trade when the throughput matters.

## 11. Key takeaways

- **vLLM V1 is a distributed system, usually on one box.** Five layers — API server, `AsyncLLM`/client, `EngineCore`, executor, workers — spread across at least two process boundaries. Name the layer, process, and plane for a symptom and you have already halved your debugging search.
- **The multiprocess design exists to beat the GIL and blocking NCCL.** One process per GPU recovers the factor-of-N that a single GIL would steal from parallel kernel launches, and lets each worker block in its own NCCL call without interpreter contention. That is the entire throughput argument, and it is why V1 reports up to ~1.7x over V0 on some models.
- **Two planes, never crossed.** Small control messages ride ZMQ DEALER sockets (frontend↔engine) and shared-memory queues `rpc_broadcast_mq` / `worker_response_mq` (executor↔workers). Large tensors ride NCCL collectives directly between worker GPUs. Using the wrong channel for either is a category error.
- **The executor is broadcast-then-gather.** `MultiProcExecutor.execute_model()` enqueues one work item to all `world_size` workers and reads the result from rank 0 alone, because after the all-reduce every rank holds the same output. `collective_rpc()` gathers from all ranks for small per-rank setup results.
- **The load-balancing score is `len(waiting) * 4 + len(running)`, lowest wins.** The four-to-one weight steers new requests toward engines that can start them now, protecting TTFT under load.
- **A worker boots in three phases: init device, load model, init KV cache.** The KV phase runs a profiling forward pass to size the block pool, which is why `gpu_memory_utilization` is a direct knob on capacity and OOM risk, and why one crowded GPU caps the whole TP group via the `min()` over per-rank block counts.
- **Executor choice follows topology.** One GPU → `UniProcExecutor`. Multiple GPUs, one node → `MultiProcExecutor` (`mp`, the default). Multiple nodes → Ray. Never go multi-node until a single node truly cannot hold the model.
- **At TP=4, NCCL is your communication budget, not the control plane.** The collectives cost single-digit milliseconds per token; the shared-memory queues cost microseconds. Optimize the interconnect and the TP degree, not the message bus.
- **Every process boundary is a failure surface.** GPUs idle → control plane above the workers. GPUs pegged and hung → data-plane collective. Hang before serving → NCCL rendezvous in worker boot. The anatomy is a triage decision tree.

## Further reading

- **["Anatomy of vLLM"](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)** — the vLLM team's own walkthrough of the request path, the `DPLBAsyncMPClient` load-balancing score, `EngineCoreProc`'s three threads, and the `MultiProcExecutor` broadcast/gather. The primary source for every class name in this post.
- **[vLLM V1 engine design and alpha announcement](https://docs.vllm.ai/)** — the rationale for the multiprocess rewrite, the V0→V1 throughput deltas, and the `VLLM_USE_V1` / `VLLM_ENABLE_V1_MULTIPROCESSING` switches.
- **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (Shoeybi, Patwary, Puri, LeGresley, Casper, Catanzaro, 2019) — the origin of the tensor-parallel sharding, and the two-all-reduces-per-block cost that the data plane pays at serving time.
- **[vLLM deep dive: architecture, APIs, and production operations](/blog/machine-learning/model-serving/vllm-deep-dive)** — `EngineArgs`, prefix caching, chunked prefill, speculative decoding, and quantization on top of the distributed skeleton in this post.
- **[Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)** — what `engine_core.step()` actually decides on each iteration of the busy loop.
- **[Tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving)** — when each parallelism strategy earns its collective overhead, and how they compose.
- **[Multi-node LLM serving for 100B+ models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus)** — TP-within-node, PP-across-node, and the Ray executor backend in practice.
- **[Running vLLM distributed in production](/blog/machine-learning/model-serving/running-vllm-distributed-in-production)** and **[Debugging vLLM distributed serving](/blog/machine-learning/model-serving/debugging-vllm-distributed-serving)** — the operational and failure-mode companions that build runbooks on this anatomy.
