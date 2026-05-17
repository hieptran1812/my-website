---
title: "CUDA Graphs: Killing Launch Overhead in GPU Workloads"
publishDate: "2026-05-17"
category: "machine-learning"
subcategory: "Deep Learning"
tags:
  [
    "cuda",
    "cuda-graphs",
    "gpu",
    "performance-optimization",
    "pytorch",
    "deep-learning",
    "inference",
    "kernel-launch",
    "llm-serving",
  ]
date: "2026-05-17"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

## Why your GPU is starving

Here is a result that surprises almost every engineer the first time they profile a small model: a GPU that the dashboard reports as "100% utilized" can still be idle more than a third of the time. The utilization counter you are watching does not mean what you think it means. On NVIDIA hardware, `nvidia-smi`'s utilization figure is the percentage of the *sampling window* in which *at least one* kernel was resident — not the fraction of SM cycles doing useful arithmetic. A GPU that runs a 4-microsecond kernel, sits idle for 6 microseconds, runs another 4-microsecond kernel, and repeats will happily report "100% utilization" while wasting 60% of its wall-clock time.

That idle time has a name, and the name is **launch overhead**. Every time the CPU tells the GPU to run a kernel, it pays a fixed tax: a runtime API call, a driver thunk, argument marshalling, and a push onto a hardware command queue. On a modern machine that tax is somewhere between 3 and 10 microseconds of pure CPU work — and crucially, it is *the same* whether the kernel that follows takes 2 microseconds or 2 milliseconds. For a 70B-parameter matmul that tax is a rounding error. For the elementwise add in a LayerNorm, the tax is *larger than the kernel itself*.

Modern deep learning models are not made of a few enormous kernels. They are made of *thousands* of small ones. A single transformer decoder layer at batch size 1 dispatches on the order of a hundred kernels: projections, attention scores, softmax, the residual adds, the LayerNorms, the activation functions, the elementwise scaling. Multiply by 32 or 80 layers and a single forward pass is several thousand kernel launches. At 5 microseconds of CPU overhead each, that is 15–25 milliseconds of pure dispatch cost before a single FLOP of *useful* arithmetic that the CPU was too slow to feed is even counted. For an LLM generating tokens one at a time, this overhead is not a tail concern. It is frequently *the* concern.

CUDA Graphs are the mechanism NVIDIA built to delete that tax. The idea is deceptively simple: instead of having the CPU describe the work kernel-by-kernel every iteration, you describe the work *once*, freeze it into an immutable graph object, and then replay the entire graph with a single API call. The CPU stops being a bottleneck because the CPU stops doing per-kernel work. This article is a deep, opinionated tour of how that mechanism actually works, where it quietly breaks, and how to wield it in production without shipping a corruption bug.

### Why CUDA Graphs are different

| Common assumption | The naive mental model | The reality |
|---|---|---|
| "My model is GPU-bound, so the CPU does not matter." | The CPU just kicks off work and waits. | The CPU *serializes* every kernel launch; a slow launch loop directly stalls the GPU. |
| "100% GPU utilization means the GPU is saturated." | Utilization = useful work. | Utilization = "a kernel was resident", which includes tiny kernels separated by idle gaps. |
| "CUDA Graphs make my kernels faster." | Graphs optimize the GPU code. | Graphs change *nothing* about kernel arithmetic; they only remove CPU-side dispatch overhead. |
| "I can just turn graphs on with a flag." | It is a free switch. | Graphs impose a hard contract: static shapes, static addresses, no host sync inside the captured region. |
| "Capture records what my kernels compute." | The graph stores logic. | The graph stores *pointers and parameters as literal values* — replay reuses the exact addresses captured. |

The through-line of that table: CUDA Graphs are a *systems* optimization, not a *kernel* optimization. They do not touch your arithmetic. They attack the gap *between* your arithmetic. Get that framing right and everything else in this article follows.

## The mental model

![Per-kernel launch versus graph replay: launch overhead leaves the GPU idle between short kernels, while replay submits the whole sequence at once](/imgs/blogs/cuda-graph-1.png)

The diagram above is the mental model, and the rest of this article is a tour of it. On the left is the world without graphs: the CPU issues `cudaLaunchKernel` for K1, the GPU runs K1, and then the GPU *sits idle* while the CPU grinds through the dispatch path for K2. The wall-clock time is the sum of kernel time *plus every CPU gap between kernels*. On the right is the world with a graph: the CPU makes one `cudaGraphLaunch` call, the GPU receives the entire pre-validated sequence at once, and the kernels run back-to-back with no bubbles. The wall-clock time collapses toward the sum of kernel time alone.

Two things are worth internalizing before we go deeper. First, **the kernels are identical in both worlds.** K1, K2, K3 contain the same SASS, touch the same memory, and take the same number of GPU cycles. Graphs do not make K1 faster. They make K2 *start sooner*. Second, **the win is bounded by the size of the gaps.** If your kernels are long enough that the CPU comfortably stays ahead of the GPU, there are no gaps to remove, and graphs do nothing. We will quantify that crossover precisely in the final section. For now, hold the picture: graphs pack a pre-described sequence, and the payoff is exactly the idle time you delete.

## 1. The launch-overhead tax

> A kernel launch is not free, it is not nearly free, and on small models it is the single most expensive thing your code does that is not arithmetic.

![Anatomy of a single kernel launch: every cudaLaunchKernel call walks a fixed CPU-side stack whose cost is independent of kernel size](/imgs/blogs/cuda-graph-5.png)

When you call `cudaLaunchKernel` — or the `<<<grid, block>>>` syntax that desugars into it, or the PyTorch dispatcher that eventually does — you trigger a fixed sequence of CPU-side stages, shown in the figure above. Walk them top to bottom:

1. **The runtime API entry.** The CUDA Runtime (`libcudart`) receives the call, looks up the current device and stream context, and resolves the kernel's function handle. Cheap, but not zero — roughly half a microsecond.
2. **The driver thunk and ioctl validation.** The runtime calls into the user-mode driver (`libcuda`), which validates the launch configuration — grid and block dimensions within device limits, shared-memory request within the per-block budget, the stream still alive. This is the most variable stage; on a loaded system with driver contention it can balloon.
3. **Argument marshalling.** Every kernel argument — every pointer, every scalar — is copied into a launch parameter buffer the GPU can read. A kernel with 12 arguments costs more here than one with 3.
4. **Command queue push.** The launch command, now fully formed, is written into the GPU's hardware command queue (the "channel"). This is the handoff point: once the command is in the queue, the GPU's front-end can pick it up asynchronously.
5. **GPU front-end dispatch.** The GPU's grid management unit pulls the command, allocates the thread blocks to SMs, and execution finally begins.

Stages 1 through 4 are *CPU work*, and they run on the thread that issued the launch. That is the tax. On a current-generation server CPU with a warm cache, the total lands around 3–5 microseconds; on a slower core, under driver contention, or with many kernel arguments, 8–10 microseconds is routine. The number you should burn into memory is **roughly 5 microseconds per launch**, and **it does not shrink as your kernel does.**

Here is how to measure it on your own hardware rather than trusting my number. The cleanest tool is Nsight Systems:

```bash
  # Profile 200 iterations of a training step, capturing CUDA API + GPU activity.
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --sample=cpu \
  --output=launch_overhead \
  --force-overwrite=true \
  python train_step.py --iters 200 --batch-size 1

  # Then summarize: how much wall time went into the launch API itself?
nsys stats --report cuda_api_sum launch_overhead.nsys-rep
```

The `cuda_api_sum` report breaks down time spent inside each CUDA API call. If `cudaLaunchKernel` shows up with a total duration that is a double-digit percentage of your step time, you have found launch-bound code, and CUDA Graphs are very likely the highest-leverage fix available. The `--cuda-graph-trace=node` flag matters once you *do* adopt graphs — without it, Nsight reports a captured graph as one opaque blob; with it, you see each node.

A worked example makes the stakes concrete. Take that 7B-parameter LLM decoding one token at a time. A single decode step issues, conservatively, 300 kernels — 32 layers, roughly 9–10 kernels per layer for the projections, attention, norms, and residuals. Suppose each kernel does about 25 µs of genuine GPU work. The *ideal* step time is then $300 \times 25\,\mu s = 7.5\,\text{ms}$. Now add launch overhead at 5 µs per kernel. If the CPU cannot stay ahead — and at 5 µs of CPU work per 25 µs kernel, it is on a knife's edge — the GPU stalls waiting for the next launch, and the realized step time creeps toward $300 \times (25 + 5)\,\mu s = 9.0\,\text{ms}$. That 1.5 ms gap is 20% of your latency, it buys nothing, and it scales with model depth. Capture the step as a graph and the 300 launches collapse into one: the CPU issues a single `cudaGraphLaunch`, never falls behind, and the step time returns to the 7.5 ms the kernels actually need. The arithmetic did not change. The bubbles disappeared. That is the entire value proposition in one calculation, and it is why every production LLM serving stack graphs its decode loop.

A second-order point that trips people up: **the launch tax is per-stream-serialized, not per-core-parallelized.** You might think "I have 32 CPU cores, I will launch kernels from 32 threads." But kernels submitted to the *same* CUDA stream must be ordered, and the driver serializes them. Multi-threading your launch loop only helps if you also have independent streams, and even then the driver has internal locks. The honest fix for launch overhead is not "launch faster in parallel" — it is "launch less", which is precisely what graphs do.

### Second-order optimization: the overhead is amortized, not eliminated

A subtlety worth stating plainly: a CUDA Graph does not make launch overhead *vanish*. It *amortizes* it. Capturing the graph still pays the per-kernel cost once, during capture. Instantiating still does CPU work. What changes is that the *replay* — the thing on your hot path, the thing you do thousands of times — costs one launch instead of N. So the real claim is "you pay the tax once instead of every iteration", and the break-even point is "the graph is replayed enough times to amortize capture and instantiation". For a training loop that runs a million steps, that is trivially satisfied. For a graph you build and replay twice, it is not. Keep the amortization horizon in mind; it reappears in several case studies below.

## 2. Graphs as a DAG

> A stream is a list. A graph is a dependency structure. The runtime can optimize a structure in ways it can never optimize a list.

![A CUDA Graph is a DAG of operation nodes: nodes are operations and edges are dependencies the runtime could never infer from a flat stream](/imgs/blogs/cuda-graph-2.png)

A CUDA stream is, semantically, a queue: operations enqueued on a stream execute in issue order, full stop. That ordering is *total* — even when two adjacent kernels are completely independent, the stream forces one to wait for the other. The stream model is simple, which is its virtue and its limit. It cannot express "these two kernels may run concurrently" without you manually splitting work across multiple streams and managing events by hand.

A CUDA Graph is a different data structure: a **directed acyclic graph** of nodes, where nodes are operations and edges are dependencies. The figure above shows the shape. The node types you can place in a graph are:

| Node type | What it does | API constructor |
|---|---|---|
| Kernel | Launches a GPU kernel | `cudaGraphAddKernelNode` |
| Memcpy | Host↔device or device↔device copy | `cudaGraphAddMemcpyNode` |
| Memset | Fills a buffer with a value | `cudaGraphAddMemsetNode` |
| Host | Runs a CPU callback | `cudaGraphAddHostNode` |
| Child graph | Embeds another graph as a sub-routine | `cudaGraphAddChildGraphNode` |
| Event record / wait | Cross-stream synchronization points | `cudaGraphAddEventRecordNode` |
| Memory alloc / free | Graph-owned allocations (CUDA 11.4+) | `cudaGraphAddMemAllocNode` |

The dependency edges are what make a graph more than a fancy list. Because the runtime sees the *entire* dependency structure before execution begins, it can do two things a stream forbids. First, it can **run independent nodes concurrently** without you splitting streams manually — if `memset` and `kernel A` have no edge between them, they may overlap. Second, and this is the real performance lever, it can **schedule the whole thing with the dependency structure resolved ahead of time.** With a stream, the driver discovers "what runs next" only as each operation completes. With a graph, "what runs next" was computed at instantiation, so the GPU's front-end can dispatch the next node the instant its predecessors retire — no CPU round-trip, no driver involvement.

There is one rule the name itself enforces: a graph is **acyclic**. You cannot express "kernel A, then kernel B, then back to kernel A" as a cycle. If you want to repeat work, you replay the whole graph, or you build a graph whose body *is* the loop body and call `cudaGraphLaunch` in a host-side loop. This is not a limitation in practice — almost every workload that benefits from graphs is an iterated *body*, and the iteration lives outside the graph.

You build a graph one of two ways. You can construct it **explicitly**, calling `cudaGraphAddKernelNode` and friends and wiring up dependency arrays by hand. Here is what that looks like for a trivial two-node graph, just so the shape is concrete:

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Node A: a kernel with no dependencies.
cudaGraphNode_t nodeA;
cudaKernelNodeParams paramsA = {};
paramsA.func          = (void*)kernel_A;
paramsA.gridDim       = dim3(256);
paramsA.blockDim      = dim3(256);
void* argsA[]         = { &d_in, &d_mid };
paramsA.kernelParams  = argsA;
cudaGraphAddKernelNode(&nodeA, graph, /*deps=*/nullptr, /*numDeps=*/0, &paramsA);

// Node B: depends on A. The dependency array IS the edge.
cudaGraphNode_t nodeB;
cudaKernelNodeParams paramsB = {};
paramsB.func          = (void*)kernel_B;
paramsB.gridDim       = dim3(256);
paramsB.blockDim      = dim3(256);
void* argsB[]         = { &d_mid, &d_out };
paramsB.kernelParams  = argsB;
cudaGraphNode_t depsB[] = { nodeA };
cudaGraphAddKernelNode(&nodeB, graph, depsB, /*numDeps=*/1, &paramsB);
```

That is twenty lines for *two* nodes. Transcribing a hundred-kernel transformer forward pass this way is miserable, error-prone, and gives you nothing capture does not — so almost nobody does it for real models. Its one genuine virtue is that you get node *handles* (`nodeA`, `nodeB`) for free, which makes fine-grained `cudaGraphExecUpdate` straightforward; with capture you have to fish those handles out afterward. The other way — **stream capture** — is what everyone actually uses, and it gets its own section.

## 3. The three-phase lifecycle

> Build once, instantiate once, replay forever. If you are doing any of the first two on your hot path, you have misunderstood the tool.

![The three-phase CUDA Graph lifecycle: define and instantiate are paid once, only launch sits on the per-iteration hot path](/imgs/blogs/cuda-graph-3.png)

A CUDA Graph passes through three distinct phases, and understanding which phase costs what is the difference between a 35% speedup and a regression.

**Phase 1 — Define.** You produce a `cudaGraph_t`, the *template*. This is a topology description: nodes, edges, parameters. It is not executable. You get one either by explicit construction or, far more commonly, by stream capture. The cost here is roughly the cost of doing the work once without a graph — capture still issues every kernel launch through the normal path; it just records instead of executing.

**Phase 2 — Instantiate.** You call `cudaGraphInstantiate(&execGraph, graph, ...)` to turn the template `cudaGraph_t` into an executable `cudaGraphExec_t`. This is the expensive, underappreciated step. Instantiation walks the entire DAG, validates it, resolves the topological order, allocates the internal data structures the GPU front-end will consume, and bakes the kernel parameters into a launch-ready form. Instantiation of a realistic model graph costs **tens to low hundreds of microseconds** — sometimes more. It is emphatically not something you do every iteration.

**Phase 3 — Launch (replay).** You call `cudaGraphLaunch(execGraph, stream)`. *This* is the hot path, and this is where the magic lives. A graph launch submits the entire pre-instantiated DAG to the stream with a single driver interaction. The CPU cost is roughly one kernel launch's worth of overhead — call it 1–2 microseconds — *regardless of how many nodes the graph contains*. A graph with 2,000 kernel nodes launches for the same CPU cost as a graph with 5.

The figure above draws the asymmetry explicitly: define and instantiate carry "once" labels; launch carries "N times". The entire performance argument for CUDA Graphs reduces to this asymmetry. You move per-kernel CPU work out of the inner loop (phase 3) and into a one-time setup (phases 1 and 2). If your workload is iterated — a training loop, a decode loop, a fixed inference pipeline serving the same shape — the amortization is overwhelming. If your workload is not iterated, there is nothing to amortize, and you should not be using graphs.

Here is the lifecycle in code, deliberately written to make the phase boundaries obvious:

```cpp
#include <cuda_runtime.h>

cudaGraph_t      graph;       // phase 1 output: the template
cudaGraphExec_t  execGraph;   // phase 2 output: the executable
cudaStream_t     stream;
cudaStreamCreate(&stream);

// ---- Phase 1: DEFINE (once) ----
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
my_workload(stream);          // issues all the kernel launches; they are recorded
cudaStreamEndCapture(stream, &graph);

// ---- Phase 2: INSTANTIATE (once) ----
cudaGraphInstantiate(&execGraph, graph, /*flags=*/0);

// ---- Phase 3: LAUNCH (the hot path, N times) ----
for (int step = 0; step < num_steps; ++step) {
    cudaGraphLaunch(execGraph, stream);
}
cudaStreamSynchronize(stream);

// Cleanup (once)
cudaGraphExecDestroy(execGraph);
cudaGraphDestroy(graph);
```

Notice the structure: two setup calls, then a tight loop that calls exactly one CUDA API per iteration. That loop is the payoff. The CPU thread spends nearly all its time *not* talking to the driver, which means it never falls behind the GPU, which means the GPU never waits.

### Second-order optimization: instantiation flags

`cudaGraphInstantiate` takes a flags argument that most code passes as `0` and never revisits. Two flags are worth knowing. `cudaGraphInstantiateFlagAutoFreeOnLaunch` makes the executable graph automatically free any graph-owned allocations at the start of each launch — useful when a graph contains alloc/free nodes and you replay it in a loop. `cudaGraphInstantiateFlagUseNodePriority` honors per-node stream priorities during scheduling. There is also `cudaGraphInstantiateFlagDeviceLaunch`, which prepares the graph to be launched *from the device itself* — a kernel inside the graph can re-launch the graph without any CPU involvement at all. Device-side graph launch is the logical endpoint of the whole launch-overhead story: not just one CPU call per iteration, but *zero*. It is advanced, it has real constraints, and most workloads do not need it — but it is worth knowing the ceiling exists. For ordinary application code, `0` is the right flag, and the lifecycle above is the whole API surface you touch.

## 4. Stream capture

> Stream capture is a tape recorder. You press record, run your normal code, press stop, and you get back a graph of everything that would have happened.

![Stream capture records work instead of running it: between Begin and End capture the launches are recorded into a graph, not executed on the GPU](/imgs/blogs/cuda-graph-4.png)

Nobody hand-builds graphs for real models. They use **stream capture**, which is the single most important API in this whole story. The premise: you bracket a region of normal CUDA code with `cudaStreamBeginCapture` and `cudaStreamEndCapture`, and every operation issued to the captured stream in between is *recorded into a graph instead of executed*. The figure above shows the shape — launches go in, nothing runs on the GPU, a `cudaGraph_t` comes out.

The minimal pattern:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

// Everything below is RECORDED, not run. The GPU executes nothing.
launch_layernorm<<<g, b, 0, stream>>>(x, gamma, beta, out);
launch_gemm<<<g, b, 0, stream>>>(out, w, h);
launch_gelu<<<g, b, 0, stream>>>(h, h);
cudaMemcpyAsync(d_result, h, n * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);

cudaStreamEndCapture(stream, &graph);   // graph now holds the recorded DAG
```

Three things about this code are non-obvious and important.

**Capture builds the dependency edges from the stream and event structure.** When you capture a single stream, you get a linear chain — each operation depends on the previous one, because that is what the stream's ordering semantics demand. To get *parallelism* in the captured graph you must, during capture, fork work onto additional streams and join with events. The capture machinery translates your stream/event choreography into graph edges. We cover this in the next section.

**The capture mode matters.** `cudaStreamBeginCapture` takes a mode argument:

| Mode | Meaning | When to use |
|---|---|---|
| `cudaStreamCaptureModeGlobal` | Any unsafe CUDA call from *any* thread aborts the capture | The safe default for single-threaded code |
| `cudaStreamCaptureModeThreadLocal` | Only unsafe calls from the *capturing thread* abort capture | Multi-threaded apps where other threads do unrelated CUDA work |
| `cudaStreamCaptureModeRelaxed` | Unsafe calls are not policed | Escape hatch; you are promising you know what you are doing |

PyTorch uses `ThreadLocal`. Most hand-written capture should too, unless you have a single-threaded program, in which case `Global` is fine and catches the most bugs.

**Certain calls are illegal during capture and will abort it.** Anything that synchronizes the host with the device — `cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaMemcpy` (the synchronous variant), or a CPU read of a GPU result like a `.item()` in PyTorch — is forbidden inside a capture region. The reason is structural: capture records a *description* of work; it does not run it; so there is no result to synchronize on or read back. A synchronous call inside capture is a logical contradiction, and the runtime resolves the contradiction by failing the capture with `cudaErrorStreamCaptureUnsupported`. Case study 5 below is an entire incident born from one stray `.item()`.

### The warmup requirement

There is a requirement that the API does not enforce but production correctness does: **you must warm up before you capture.** Run the exact workload you are about to capture a few times — typically 3 to 11 iterations — *outside* the capture region, on a side stream, first.

Why? Because the first execution of a CUDA workload does a great deal of *lazy* one-time work that you do not want frozen into your graph or, worse, attempted *during* capture where it may be illegal:

- **cuBLAS and cuDNN algorithm selection.** The first matmul of a given shape triggers a heuristic search or autotuning pass that picks an algorithm. If you capture before that selection has happened, you may capture the *autotuning itself* (which can do host syncs and abort your capture) or bake in a cold-cache algorithm choice.
- **Lazy module loading.** CUDA loads kernel binaries lazily on first use. First touch of a kernel is a load; you do not want that inside capture.
- **Memory allocator warmup.** The caching allocator needs to have allocated the blocks your workload uses, so that during capture no *new* `cudaMalloc` happens (raw `cudaMalloc` is a host-synchronizing call and illegal in capture).
- **NCCL communicator setup.** Collectives lazily establish channels on first call.

Skip the warmup and you get one of two failure modes: a capture that aborts with a cryptic error, or — far nastier — a capture that *succeeds* but freezes in a cold-path decision. Case study 1 is exactly the second kind.

## 5. Multi-stream graphs

> A single-stream capture gives you a chain. If your workload has parallelism, you have to choreograph it during capture, and the graph remembers.

![Multi-stream graphs capture fork and join: forking onto side streams during capture turns cross-stream events into real dependency edges](/imgs/blogs/cuda-graph-6.png)

Capture a single stream and you get a linear graph — every node depends on its predecessor. That is correct but it leaves performance on the table whenever your workload has genuinely independent work. The figure above shows the pattern that fixes it: during capture, you *fork* from the main captured stream onto side streams, and *join* back, using CUDA events as the synchronization primitive. The capture machinery translates that fork/join choreography into real dependency edges in the graph.

The mechanics are precise. To fork, you record an event on the main stream and have a side stream wait on it; that side stream is now part of the same capture. To join, the side stream records an event and the main stream waits on it. Concretely:

```cpp
cudaStream_t main, sideA, sideB;
cudaEvent_t  forkA, forkB, joinA, joinB;
// ... create all of the above ...

cudaStreamBeginCapture(main, cudaStreamCaptureModeThreadLocal);

// Fork: make sideA and sideB join this capture.
cudaEventRecord(forkA, main);
cudaStreamWaitEvent(sideA, forkA, 0);   // sideA now part of the capture
cudaEventRecord(forkB, main);
cudaStreamWaitEvent(sideB, forkB, 0);   // sideB now part of the capture

kernel_A<<<g, b, 0, sideA>>>(/* ... */);   // independent work
kernel_B<<<g, b, 0, sideB>>>(/* ... */);   // independent work

// Join: make main wait for both side streams.
cudaEventRecord(joinA, sideA);
cudaStreamWaitEvent(main, joinA, 0);
cudaEventRecord(joinB, sideB);
cudaStreamWaitEvent(main, joinB, 0);

kernel_C<<<g, b, 0, main>>>(/* ... */);    // depends on A and B

cudaStreamEndCapture(main, &graph);
```

The resulting graph has `kernel_A` and `kernel_B` with no edge between them — they are siblings, free to run concurrently — and `kernel_C` with edges from both. At replay time the GPU front-end sees that A and B are independent and can co-schedule them on free SMs, exactly as the figure shows the fork and join.

It is worth being precise about what the events become. During normal execution, `cudaEventRecord` and `cudaStreamWaitEvent` are runtime synchronization primitives — one stream literally waits on another at execution time. During *capture*, they are not executed; they are *interpreted* as topology. A `WaitEvent` on a side stream tells the capture machinery "this side stream's subsequent work depends on whatever produced that event", and the machinery emits a dependency edge accordingly. The events themselves do not survive into the graph as nodes you can see — they are consumed by the capture and turned into edges. This is why the fork/join code reads like synchronization but produces a graph with *concurrency*: you write what looks like blocking event waits, and the capture distills them into a pure dependency structure that the GPU front-end is then free to schedule as parallel as the edges allow.

A rule that is easy to get wrong: **the capture must end on the same stream it began on, and all forked streams must rejoin before `cudaStreamEndCapture`.** If a side stream is still "in" the capture when you call `EndCapture` on the main stream, the runtime fails with `cudaErrorStreamCaptureUnjoined`. Every fork needs a matching join. The discipline is the same as balancing parentheses.

This matters more than it first appears for two production patterns. The first is **compute/communication overlap** in distributed training — you want an all-reduce on a communication stream overlapping with a backward-pass kernel on a compute stream, and both inside one captured graph. The second is **multi-branch models** where two sub-networks process the same input independently before a fusion layer. In both cases, single-stream capture would serialize work that the hardware could run concurrently, and you would leave a real speedup uncaptured. Case study 4 is a cautionary tale about getting the communication stream wrong.

## 6. Updating a graph without rebuilding

> Re-instantiating a graph every iteration to change one number is like recompiling your program every time you change a config value. The fix has a name: `cudaGraphExecUpdate`.

![Re-instantiate versus cudaGraphExecUpdate: updating an exec graph in place skips topology re-validation that re-instantiation pays every time](/imgs/blogs/cuda-graph-7.png)

The static contract of CUDA Graphs — fixed shapes, fixed addresses — runs into an obvious objection: real workloads change. A training step uses a new learning rate after a scheduler tick. An inference server gets a request with a different sequence length. If "change anything" means "rebuild and re-instantiate the graph", and instantiation costs 50–100 microseconds, then a workload that changes every step has erased the entire benefit of graphs. The figure above contrasts the two paths.

NVIDIA's answer is `cudaGraphExecUpdate`. The insight: if the graph's **topology** is unchanged — same nodes, same edges, same kernels — and only **parameters** changed — a pointer, a scalar, a grid dimension — then you do not need to re-instantiate. You can patch the existing `cudaGraphExec_t` in place. The update path skips the expensive topology validation and order resolution; it only rewrites the affected node parameters. Where instantiation costs tens to hundreds of microseconds, an update costs single-digit microseconds.

There are two ways to use it. The coarse-grained way: capture a *fresh* `cudaGraph_t` with the new parameters, then ask the runtime to diff it against the existing executable:

```cpp
// Re-capture with new parameters into a fresh template graph.
cudaGraph_t newGraph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
my_workload(stream, new_params);   // same kernels, new scalars/pointers
cudaStreamEndCapture(stream, &newGraph);

// Try to patch the existing executable in place.
cudaGraphExecUpdateResultInfo info;
cudaError_t st = cudaGraphExecUpdate(execGraph, newGraph, &info);

if (st != cudaSuccess) {
    // Topology actually changed — fall back to a full re-instantiate.
    cudaGraphExecDestroy(execGraph);
    cudaGraphInstantiate(&execGraph, newGraph, 0);
}
cudaGraphDestroy(newGraph);
```

The fine-grained way: skip the re-capture and call the targeted setters directly — `cudaGraphExecKernelNodeSetParams`, `cudaGraphExecMemcpyNodeSetParams`, and so on — passing the node handle and the new parameter struct. This is the cheapest possible update because it touches exactly one node, but it requires you to have kept handles to the nodes you intend to mutate, which means explicit construction or careful bookkeeping during capture.

The critical caveat is in the `if` branch above: **`cudaGraphExecUpdate` fails if the topology changed.** Add a node, remove an edge, change a kernel function, change a grid dimension beyond what the node was instantiated to allow — any of those returns an error code, and you must fall back to a full re-instantiate. The update path is for the case where the *shape of the computation is identical* and only the *numbers* moved. Learning-rate changes, new input buffer addresses, updated loss-scale factors — all fine. A different number of layers — not fine. Case study 7 is the cost of forgetting this distinction.

## 7. The static-memory contract

> A CUDA Graph does not store "read the input tensor". It stores "read address `0x7f3a...a00`". If that address no longer holds your input, replay reads garbage — silently.

![The static-memory contract: replay reuses captured pointer values verbatim, so a reallocated buffer turns into a dangling write](/imgs/blogs/cuda-graph-8.png)

This is the section that, if you skip it, will eventually cost you a multi-day debugging session and possibly a wrong model. Read it twice.

When you capture a kernel launch, the graph records that kernel's *arguments* — and pointer arguments are recorded **as literal 64-bit address values.** The graph node for `gemm(A, B, C)` does not store "the tensor named A". It stores the number `0x7f3a8c00a000`, whatever the address of A happened to be at capture time. At replay, the kernel runs with that exact number. The figure above is the whole contract in one picture: the kernel node holds a pointer; if the buffer at that address is the same static buffer, replay is correct; if that buffer was freed and the memory reused for something else, replay writes into whatever now lives there.

The consequence is a hard rule: **every buffer a captured graph reads or writes must keep the same address for the entire lifetime of the executable graph.** This includes input buffers, output buffers, weights, and every intermediate activation. You cannot allocate a fresh tensor each iteration and expect replay to pick it up. Replay does not "pick up" anything — it reuses the captured addresses verbatim.

There are two correct ways to satisfy the contract:

**Static buffers.** Allocate every input, output, and intermediate *once*, before capture, and reuse the same allocations forever. New data does not get a new buffer — it gets *copied into* the existing one. The pattern is:

```cpp
// Allocate the static I/O buffers ONCE.
float *static_input, *static_output;
cudaMalloc(&static_input,  n * sizeof(float));
cudaMalloc(&static_output, m * sizeof(float));

// Capture a graph that reads static_input and writes static_output.
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
my_model(static_input, static_output, stream);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&execGraph, graph, 0);

// Hot loop: copy new data IN, replay, copy results OUT. Addresses never move.
for (auto& batch : batches) {
    cudaMemcpyAsync(static_input, batch.data, n * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaGraphLaunch(execGraph, stream);
    cudaMemcpyAsync(batch.result, static_output, m * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}
```

The mental shift: with graphs, buffers are *fixed slots*, and the data flows *through* the slots. You do not hand the graph a new tensor; you refill the slot the graph already knows about.

**Graph-owned allocations.** Since CUDA 11.4, a graph can contain memory-allocation and memory-free nodes (`cudaGraphAddMemAllocNode` / `cudaGraphAddMemFreeNode`). Allocations made this way are managed by the graph itself, and the runtime guarantees their addresses are stable across replays of that graph. This is more advanced and mostly relevant to library authors; for application code, static buffers are simpler and just as fast.

The reason this contract is so dangerous in practice is the *silence* of the failure. If you violate it — if some buffer the graph captured gets freed and its memory recycled — replay does not crash with a null-pointer fault. The address is still a valid GPU address; it just points at *different data now*. The kernel runs, reads or writes that memory, and produces a wrong answer with no error code. In a training loop you see it as a loss curve that quietly diverges. In inference you see it as occasionally garbled outputs. There is no exception to catch. This is why the PyTorch integration goes to such lengths around the allocator, which is the entire subject of the next section, and why case study 2 is the most expensive bug in this article.

## 8. CUDA Graphs in PyTorch

> You will almost never call `cudaGraphInstantiate` yourself. You will call `torch.cuda.graph` or `make_graphed_callables` or, increasingly, just `torch.compile(mode="reduce-overhead")`. But everything above is still happening underneath, and when it breaks, it breaks in those terms.

![How PyTorch wraps CUDA Graphs: warmup, capture into a private memory pool, then replay the same buffers each step](/imgs/blogs/cuda-graph-9.png)

PyTorch exposes CUDA Graphs through three layers of increasing convenience. All three do the same underlying thing — warm up, capture into a private memory pool, replay — shown in the figure above. The difference is how much of the static-memory contract they manage for you.

**Layer 1: the raw `torch.cuda.CUDAGraph` context manager.** The lowest-level handle. You manage the static input/output tensors yourself:

```python
import torch

model = build_model().cuda().eval()

  # Static I/O buffers — allocated once, addresses fixed forever.
static_input  = torch.zeros(1, 512, 4096, device="cuda")
static_output = torch.zeros(1, 512, 4096, device="cuda")

  # 1. WARMUP on a side stream — let cuBLAS/cuDNN pick algorithms,
  #    let the allocator warm up, let lazy modules load.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output.copy_(model(static_input))
torch.cuda.current_stream().wait_stream(s)

  # 2. CAPTURE — the region runs into a graph, not onto the GPU.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output.copy_(model(static_input))

  # 3. REPLAY — fill the static input, replay, read the static output.
for batch in loader:
    static_input.copy_(batch)        # new data INTO the fixed slot
    graph.replay()                   # one launch for the whole model
    consume(static_output.clone())   # copy OUT before the next replay
```

Every element of this article is visible in those 20 lines: the warmup loop on a side stream (section 4), the capture region (section 4), the static buffers (section 7), and the single-call replay (section 3). The `.clone()` at the end is not optional — `static_output` is overwritten by the next `replay()`, so you must copy the result out before reusing the slot.

**Layer 2: `torch.cuda.make_graphed_callables`.** A wrapper that takes a module (or callable) and a tuple of sample inputs, does the warmup and capture for you, and hands back a drop-in replacement that internally replays a graph. It also handles the *backward* pass — it captures both forward and backward graphs, which makes it usable inside a training loop:

```python
from torch.cuda import make_graphed_callables

sample = torch.randn(8, 4096, device="cuda")
graphed_model = make_graphed_callables(model, (sample,))

  # graphed_model now behaves like model, but forward AND backward
  # replay captured graphs. Inputs must match the sample's shape/dtype.
for batch in loader:                 # batch must be shape (8, 4096)
    loss = loss_fn(graphed_model(batch), targets)
    loss.backward()
    optimizer.step()
```

The constraint it cannot remove for you: **the inputs must match the sample shapes exactly.** Variable batch sizes break it. The standard workaround is bucketing — capture one graphed callable per shape bucket and route each input to the matching bucket. Case study 3 is what happens when you do not.

**Layer 3: `torch.compile(mode="reduce-overhead")` and CUDA Graph Trees.** The modern, recommended path. When you compile with `reduce-overhead`, the Inductor backend automatically wraps the compiled regions in CUDA Graphs. The hard problem it solves is that real models are not one straight-line capture — they have data-dependent branches, multiple distinct input shapes, and dynamic control flow. PyTorch's answer is **CUDA Graph Trees**: instead of one monolithic graph, it maintains a *tree* of graphs that share a single private memory pool, where each path through the tree corresponds to one branch of execution. The shared pool is the clever part — it lets graphs that are never *live simultaneously* reuse the same memory, so you do not pay a full static-buffer footprint per branch.

```python
model = build_model().cuda()
  # Inductor captures CUDA Graphs automatically; first calls warm + capture.
fast_model = torch.compile(model, mode="reduce-overhead")

for batch in loader:
    out = fast_model(batch)          # replays a graph from the tree
```

CUDA Graph Trees deserve one more paragraph because they are where most engineers will actually meet graphs in 2025 and beyond. The problem they solve is that a realistic compiled model is not one capture — `torch.compile` may produce several distinct compiled regions, separated by "graph breaks" wherever it hits Python control flow it cannot trace, and each region may be invoked with more than one input shape. A single monolithic graph cannot represent that. The tree structure does: each node in the tree is a captured graph, each path from the root corresponds to one concrete sequence of (region, shape) invocations the model actually took, and new paths are captured lazily the first time they are seen. The payoff of the *shared* memory pool is that two graphs on different branches of the tree — branches that, by construction, are never executing at the same time — can be assigned overlapping memory, so the total footprint is bounded by the deepest single path rather than the sum of all paths. The cost you pay is a capture stall the first time each new path is exercised, which is why the first few hundred requests to a freshly started `reduce-overhead` server are slower than steady state. Budget for that warmup ramp in your latency SLOs.

The **private memory pool** in the figure is the load-bearing concept across all three layers. PyTorch's caching allocator, during a capture, switches to a dedicated pool so that allocations made inside the captured region get stable addresses and — critically — do not get handed out to *non-graph* code that might free and recycle them. This pool is exactly the machinery that enforces the static-memory contract of section 7. When that machinery has a gap, you get case study 2.

A practical comparison of the three layers:

| Layer | API | Handles warmup | Handles backward | Handles dynamic shapes | Use when |
|---|---|---|---|---|---|
| Raw | `torch.cuda.CUDAGraph` | No | No | No | You need full control; building a serving engine |
| Callable | `make_graphed_callables` | Yes | Yes | No (bucket manually) | Fixed-shape training/inference of a sub-module |
| Compile | `torch.compile("reduce-overhead")` | Yes | Yes | Partial (graph trees) | Default choice for most models in 2025+ |

The honest recommendation for almost everyone: start with `torch.compile(mode="reduce-overhead")`. Drop to `make_graphed_callables` when you have a fixed-shape hot module and `torch.compile` is fighting you. Drop to the raw API only when you are building infrastructure — a custom inference server, a framework — where you need to own the capture lifecycle yourself.

## Cross-cutting concerns

Three concerns cut across every section above and deserve their own treatment before the case studies.

### Observability: profiling a graph is different

The moment you adopt graphs, your profiler output changes character. Without `--cuda-graph-trace=node`, Nsight Systems shows a captured graph as a *single* `cudaGraphLaunch` entry — you lose all per-kernel visibility, because from the API's perspective one call happened. Always profile graphed code with:

```bash
nsys profile --trace=cuda --cuda-graph-trace=node \
  --output=graphed python serve.py
```

With node-level tracing, each graph node reappears as its own timeline entry, and you can see whether the *kernels themselves* are now the bottleneck (which is the goal — it means launch overhead is gone). Nsight Compute, for kernel-level SASS analysis, profiles individual kernels inside graphs without special flags. The mental adjustment: before graphs you debug "why is my launch loop slow"; after graphs you debug "why is this kernel slow", which is a better problem to have.

There is a second observability shift worth anticipating. Before-and-after measurement is the only honest way to justify a graph, and the measurement has to be done correctly. Wrap the *replay loop* — not the capture, not the instantiation — in CUDA events and average over a few hundred iterations after a warmup, because the first replay still pays some one-time costs. Compare that against the same loop run eagerly. If you instead time the whole program, the capture and instantiation cost contaminates the number and you will under-report the win on a long-running job or over-report a regression on a short one. And always sanity-check that the graphed and eager paths produce *numerically identical* output before you trust the speed number — a graph that is fast because it silently skipped a kernel is not a win, it is case study 2 waiting to happen. The discipline is simple: measure the hot loop, measure it warm, and verify correctness first.

### Memory: graphs and the allocator are entangled

A captured graph pins memory. Every buffer the graph touches must stay alive and at a fixed address for the executable graph's whole lifetime, which means a graphed model has a *higher and less flexible* memory footprint than the same model run eagerly. With CUDA Graph Trees, the shared pool mitigates this — branches that are never co-live share memory — but the baseline is still "graphs trade memory flexibility for launch speed." On a GPU that is already memory-tight, that trade can be the thing that pushes you into OOM. Budget for it.

### Dynamic shapes: the eternal tax

The single biggest practical limitation of CUDA Graphs is that they are *shape-frozen*. A graph captured for sequence length 512 is valid only for sequence length 512. The three coping strategies, in increasing order of sophistication:

| Strategy | How it works | Cost |
|---|---|---|
| Bucketing | Capture one graph per shape bucket; route inputs to the nearest bucket | N× capture cost; some wasted compute from padding up |
| Padding | Always pad inputs up to one fixed maximum shape; one graph | Wasted compute on every under-full input |
| Graph trees | Let `torch.compile` capture a tree of graphs keyed by shape | Automatic, but capture happens lazily on first sight of each shape |

For LLM inference specifically, the field has largely converged on bucketing the batch dimension (a small set of fixed batch sizes) while keeping the sequence dimension handled by the attention kernel's own support for variable lengths. The takeaway: graphs and dynamism are in genuine tension, and every production graph deployment is, at some level, a strategy for managing that tension.

### Composability: child graphs and nesting

A graph can contain another graph as a single node — a **child graph node**, added with `cudaGraphAddChildGraphNode` or produced naturally when you capture a stream that itself replays a graph. This is more useful than it first sounds. It lets you build a library of reusable sub-graphs — a captured attention block, a captured MLP block — and assemble them into a larger graph without re-capturing the internals. The child graph's nodes are flattened into the parent at instantiation time, so there is no runtime cost to the nesting; it is purely an authoring and maintenance convenience.

The composability story has a sharp edge, though: **updates do not compose cleanly.** If you build a parent graph out of three child graphs and then want to `cudaGraphExecUpdate` one parameter buried in child number two, you update against a freshly captured *parent*, not the child in isolation — the executable graph is a flat structure and knows nothing about your authoring-time nesting. In practice teams that lean hard on child graphs end up keeping the child `cudaGraph_t` templates around precisely so they can re-assemble and re-capture the parent when a sub-block's parameters move. Child graphs are a clean way to *organize* capture; they are not a way to *partially update* an executable. Keep that distinction and nesting stays a convenience rather than a trap.

## Case studies from production

What follows are seven incidents — some I have personally debugged, some are composites of patterns common enough to be archetypes. Each is concrete because the lessons only land when concrete.

### 1. The warmup that wasn't

A team graphed the forward pass of a vision transformer for an inference service and saw the expected latency drop — about 30% — in their benchmark harness. In production, latency was *worse* than eager mode by a few percent, and wildly inconsistent. The benchmark and production differed in one way: the benchmark ran the model 50 times before measuring; production captured the graph on the very first request.

The root cause was cuBLAS algorithm selection. cuBLAS picks a matmul algorithm via a heuristic on first sight of a given problem shape, and that selection can involve a brief autotuning pass. When the graph was captured before any warmup, one of two things happened: either the capture aborted (and the service silently fell back to eager mode), or — the worse case — the capture *succeeded* but recorded a cold-path, conservative algorithm choice, freezing a slow GEMM into the graph forever. The benchmark's 50 warmup iterations had let cuBLAS settle on its fast algorithm before capture; production never did.

The fix was four lines: run the model 5 times on a side stream before entering the capture region. The lesson is the one from section 4 stated as a scar: **the warmup is not a performance nicety, it is a correctness requirement.** Anything CUDA does lazily — algorithm selection, module loading, allocator warmup — must be forced to completion *before* capture, or capture will freeze a cold decision. Always warm up, and always warm up with the *exact* shapes you will capture.

### 2. The dangling activation

This is the most expensive bug in this article and the one that most justifies section 7. A team running graphed training of a mid-size transformer noticed that, every few thousand steps, the loss would spike and then recover. Not diverge — just spike, once, and continue. It happened rarely enough that it was dismissed as "data noise" for weeks.

It was not data noise. It was a violation of the static-memory contract. Their training loop allocated a small auxiliary tensor — a per-step random mask — *inside* the region they were capturing, but they had built the graph capture before the PyTorch caching allocator's graph-pool machinery was correctly engaged for that tensor. The mask's memory was, on most steps, untouched by anything else between replays. But occasionally a non-graph code path — a logging hook that built a temporary tensor — would be handed that exact freed address by the allocator, write to it, and the *next graph replay* would read the mask kernel's input from memory that now held logging garbage. One corrupted mask, one spiked loss.

The fix was to allocate the mask buffer *once*, statically, before capture, and have the graph reuse the slot — exactly the pattern in section 7. The lesson: **a graph's failure mode is silent corruption, not a crash.** There was no exception, no `cudaErrorIllegalAddress`, because the address was always *valid* — it just sometimes pointed at the wrong data. If you adopt graphs and see *occasional, recoverable* numerical weirdness, your first hypothesis should be a static-memory violation, not your data pipeline.

### 3. Dynamic batch size

An inference team graphed a recommendation model with `make_graphed_callables` and shipped it. It worked in staging. In production it threw shape-mismatch errors on a large fraction of requests, and the fallback path — eager execution — meant production was running *un-graphed* most of the time while paying the memory cost of the captured graph anyway.

The cause was obvious in hindsight: staging traffic was synthetic and uniform, all batch size 32. Production traffic had batch sizes anywhere from 1 to 64, depending on how many items the upstream service grouped. `make_graphed_callables` captures for *exactly* the sample shape; every non-32 batch missed.

The fix was bucketing: they captured graphed callables for batch sizes 1, 2, 4, 8, 16, 32, and 64, and routed each request to the smallest bucket that fit, padding up. Padding wasted some compute on under-full buckets, but a graphed batch-of-8-padded-to-16 still beat an eager batch-of-8. The lesson: **graphs are shape-frozen, and your traffic is not.** Before you graph anything, characterize the *real* distribution of input shapes, not the staging distribution, and design a bucketing strategy that covers it. A graph that misses 60% of requests is worse than no graph.

### 4. NCCL inside the graph

A team scaling training to multiple GPUs tried to capture a training step that included an all-reduce of gradients. The capture hung. Not an error — a hang, the worst kind of failure, requiring a `kill -9` and offering no stack trace.

The root cause was an interaction between NCCL collectives and stream capture. NCCL collectives are *capturable*, but only under specific conditions: the communicator must be warmed up first (the channels established), and the collective must be issued in a way consistent with capture. This team's first cut issued the all-reduce on the *default* stream while capturing a *different* stream, and the collective's internal synchronization deadlocked against the capture state.

The fix had two parts. First, warm up the NCCL communicator with a throwaway all-reduce before capture, so channel establishment did not happen during capture. Second, issue the collective on a stream properly forked into the capture (section 5's fork/join pattern) rather than on an unrelated stream. With the communication stream correctly part of the captured graph, the all-reduce became a set of nodes that overlapped cleanly with backward-pass compute. The lesson: **collectives can live in graphs, but they are the most fragile thing you will capture.** Warm the communicator, fork the comm stream into the capture explicitly, and test multi-GPU graph capture in isolation before integrating it.

### 5. CPU sync mid-capture

A developer adding graph capture to an existing model hit `cudaErrorStreamCaptureUnsupported` the moment they wrapped their forward pass. The error pointed nowhere useful — just "an unsupported operation occurred during capture." The forward pass was 600 lines across a dozen modules.

The offending line was a single `.item()` call buried in a custom loss component, used to pull a scalar off the GPU for a `logging.debug` statement. `.item()` synchronizes the host with the device — it has to, because it returns a Python number — and host synchronization is illegal inside capture, as section 4 explained. One debug log line, written months earlier by someone else, broke the entire capture.

Finding it took an afternoon of bisecting the forward pass by progressively narrowing the capture region. The fix was to guard the `.item()` behind a "not currently capturing" check (`torch.cuda.is_current_stream_capturing()`) so the debug path was simply skipped during capture. The lesson: **any host-device synchronization inside a capture region is a hard failure, and the error will not tell you where it is.** Before capturing a large existing model, audit it for the usual suspects — `.item()`, `.cpu()`, `.numpy()`, `print(tensor)`, synchronous `.to()`, any `assert` on a tensor value, any Python-side branch on a tensor's contents. Each one is a landmine.

### 6. The 200-microsecond decode step

A team serving a 7B-parameter LLM measured their per-token decode latency at about 11 milliseconds and assumed it was GPU-bound — it was an LLM, after all, and the GPU was at "100% utilization." A profile told a different story. The actual GPU kernel time for one decode step was around 7 milliseconds. The other 4 milliseconds was launch overhead: a single decode step of a 7B model at batch size 1 dispatches several hundred small kernels — the per-head attention work, the elementwise residuals, the RMSNorms, the rotary embedding application — and at roughly 5 microseconds of CPU dispatch each, several hundred launches is several milliseconds of pure overhead.

This is the textbook case for graphs, because LLM decode is the textbook launch-bound workload: small batch, small per-token compute, enormous kernel *count*, and the *exact same* sequence of kernels every single step. They captured the decode step as a graph — with the KV cache as a static buffer, which it naturally already was — and per-token latency dropped from 11 ms to about 7.3 ms, a 34% reduction, almost exactly the launch overhead they had measured. Throughput rose correspondingly.

The lesson connects to broader serving architecture: graphs and the [KV cache](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) are natural partners. The KV cache is *already* a static, fixed-address buffer that persists across decode steps — it satisfies the static-memory contract for free. And decode is *already* a fixed, repeated sequence of kernels. LLM decode is almost custom-built to be graphed, which is exactly why every serious inference engine — vLLM, TensorRT-LLM, SGLang — captures the decode step as a CUDA Graph by default. If you are serving an LLM and *not* graphing decode, you are leaving a third of your latency on the floor.

### 7. Graph update vs rebuild

A team graphed their training step and saw no speedup at all — graphed and eager were within noise of each other. The graph was capturing correctly; replay was happening. So where did the win go?

It went into `cudaGraphInstantiate`. Their training step used a learning-rate warmup schedule, so the learning rate changed every step. Their first implementation handled that the naive way: every step, re-capture the graph with the new learning rate and call `cudaGraphInstantiate` to make it executable. Instantiation cost roughly 80 microseconds per step. They had moved the per-kernel launch overhead out of the loop and then moved an *80-microsecond instantiation* right back in. Net zero.

The fix was `cudaGraphExecUpdate`, exactly as section 6 describes. The learning rate is a *parameter* — it changes the value a kernel reads, not the topology of the computation. So instead of re-instantiating, they re-captured into a fresh template graph and called `cudaGraphExecUpdate` to patch the existing executable in place, at a cost of a few microseconds. With that change the graphed step finally beat eager by the expected margin. The lesson: **instantiation is not on the hot path, and if you have put it there, you have built a slower program with extra steps.** When something changes every iteration, ask first whether it is a parameter change or a topology change. If it is a parameter change — and learning rates, input addresses, and loss scales all are — `cudaGraphExecUpdate` is the answer, and re-instantiation is a bug.

### 8. The capture that doubled memory

A team graphed inference for a model that was already close to the memory ceiling on an L40S — the eager model fit with a couple of gigabytes to spare. The graphed version OOMed on startup, before serving a single request.

The cause was the static-buffer footprint. In eager mode, the caching allocator constantly recycles memory: an activation tensor is allocated, used, freed, and its block handed to the next activation. The model's *peak* live memory was well under its *total churned* memory. Capture broke that recycling. Every buffer the graph touched had to stay alive at a fixed address for the graph's lifetime — so the graphed model's footprint was closer to the sum of all distinct activations than to the eager peak. Add the warmup that ran the model on a side stream while the capture pool was also resident, and startup briefly needed nearly twice the eager footprint.

The fix had two parts. First, they moved to `torch.compile`'s CUDA Graph Trees, whose shared memory pool lets activations that are never co-live reuse the same blocks — recovering much of the recycling that raw capture had destroyed. Second, they captured *after* warmup buffers were freed rather than overlapping the two. Footprint came back down to roughly 1.2× eager, which fit. The lesson: **graphs trade memory flexibility for launch speed, and the trade is not free.** Before graphing a memory-tight model, measure the static footprint — not the eager peak — and confirm it fits. The shared-pool machinery in graph trees exists precisely because the naive footprint is often unacceptable.

### 9. The graph that captured stale weights

A research team graphed the forward pass of a model for fast evaluation during training — every few hundred steps they would run a validation pass, and they graphed it to make validation cheap. The validation numbers were suspiciously flat: the model appeared to stop improving after the first validation, even though training loss kept dropping.

The model was improving. The graph was not seeing it. They had captured the validation graph once, early in training, and the captured kernel nodes held the *weight pointers as of capture time*. That part was fine — the weight tensors kept stable addresses. What was *not* fine is that the team, between training and validation, swapped in an EMA (exponential moving average) copy of the weights by reassigning the module's parameter tensors to *different* tensor objects at *different* addresses. The graph kept reading the original addresses. Validation ran, every time, on the weights frozen at first capture.

The fix was to make the EMA update *write into* the existing weight buffers — `ema_param.copy_(...)` rather than rebinding — so the addresses the graph captured always held current data. The lesson is section 7 from a different angle: **the static-memory contract covers weights, not just activations.** A graph reads weights by address. Anything that swaps weights by *rebinding the tensor object* — EMA, weight averaging, loading a checkpoint, LoRA adapter swapping — silently desynchronizes the graph. Update weights *in place* or re-capture. There is no third option.

### 10. The premature graph

A team added CUDA Graphs to a model that was still under active development — new layers being added, the architecture in flux week to week. They reasoned that graphing early would "bake in" the performance win and save a future migration.

It did the opposite. Every architecture change altered the graph's topology, which meant every change forced a full re-capture and re-instantiation, which meant the graph machinery had to be re-validated against every change. Worse, when a subtle numerical bug appeared, debugging it through a captured graph was agony: graphs freeze the computation, make per-kernel inspection awkward, and turn some failures silent (section 7). The team spent more engineering time fighting their own graphs than the graphs ever saved in runtime.

They ripped the graphs out, stabilized the architecture, got the model correct in eager mode, and *then* re-introduced graphs in a single afternoon once the topology stopped moving. The lesson: **graphs are a finishing optimization, not a foundation.** They reward a stable, correct, profiled workload and they punish a moving target. The right time to graph is after the architecture is frozen, the numerics are verified, and a profile has confirmed you are launch-bound. Graphing before any of those three is true is borrowing trouble against a speedup you have not yet earned.

## When to reach for CUDA Graphs — and when not to

![When CUDA Graphs actually pay off: graphs win only when launch overhead is a meaningful fraction of each kernel's runtime](/imgs/blogs/cuda-graph-10.png)

Every optimization has a domain of validity, and CUDA Graphs have a sharply defined one. The figure above draws the crossover. We can make it quantitative. Let a workload dispatch $N$ kernels, each running for $t_k$ microseconds on the GPU, each costing $t_\ell$ microseconds of CPU launch overhead. Without graphs, if the CPU is the bottleneck, wall time per iteration is roughly $N \cdot (t_k + t_\ell)$ in the worst case where the GPU fully stalls on each launch. With a graph, it is roughly $N \cdot t_k + t_g$, where $t_g$ is the single graph-launch overhead. The fractional speedup is approximately:

$$\text{speedup} \approx \frac{t_\ell}{t_k + t_\ell}$$

Read that formula. When $t_k \gg t_\ell$ — kernels far longer than the launch tax — the speedup tends to zero. When $t_k \approx t_\ell$ — kernels about as long as the tax — the speedup approaches 50%. The crossover is entirely about the ratio of kernel duration to launch overhead. With $t_\ell \approx 5\,\mu s$, kernels under ~20 µs are squarely in graph territory; kernels over ~200 µs are not.

It helps to put representative workloads on that curve rather than reasoning in the abstract. The table below is a rough field guide — the exact numbers depend on hardware and model, but the *ordering* is stable:

| Workload | Mean kernel duration | Launch-bound? | Expected graph speedup |
|---|---|---|---|
| LLM decode, batch 1, 7B | ~10–30 µs | Heavily | 25–40% step-time reduction |
| LLM decode, batch 64, 7B | ~80–200 µs | Mildly | 5–15% |
| LLM prefill, long sequence | ~500 µs+ | No | ~0%, skip it |
| CNN training, batch 256 | ~300 µs+ | No | ~0%, skip it |
| CNN training, batch 1 (RL rollout) | ~15–40 µs | Heavily | 20–35% |
| Elementwise-heavy diffusion sampler | ~5–25 µs | Heavily | 20–40% |
| Large-batch GEMM benchmark | ~1 ms+ | No | negligible |

The pattern is unmistakable: **small batch and small models are launch-bound; large batch and large models are compute-bound.** Graphs are a small-batch, small-kernel optimization. The single most common mistake is graphing a large-batch training job, seeing no speedup, and concluding "graphs do not work" — when the real conclusion is "this workload had no launch bubbles to remove in the first place."

**Reach for CUDA Graphs when:**

- Your workload is **iterated** — the same sequence of kernels runs many times. Training loops, LLM decode, fixed inference pipelines. The capture and instantiation cost must amortize over many replays.
- Your profile shows **`cudaLaunchKernel` consuming a double-digit percentage** of wall-clock time. This is the direct diagnostic; do not graph on a hunch, graph on a profile.
- You dispatch **many small kernels** — small batch sizes, elementwise-heavy models, LLM decode. High kernel count and low per-kernel duration is the exact profile graphs target.
- Your shapes are **static or bucketable into a small set.** Either truly fixed, or a handful of buckets you can capture separately.
- Your buffers **already persist** across iterations — KV caches, model weights, preallocated activation arenas. If you already satisfy the static-memory contract, adoption is nearly free.

**Skip CUDA Graphs when:**

- Your workload is **compute-bound with large kernels.** A model dominated by big GEMMs — large-batch training, large-batch inference — has no launch bubbles to remove. The speedup formula gives you near zero. Do not add the capture machinery for nothing.
- Your shapes are **genuinely, unboundedly dynamic.** If every input is a different shape and there is no sane bucketing, you will spend more time capturing graphs than replaying them.
- Your workload **runs once.** A one-shot script, a single inference call. There is nothing to amortize the capture and instantiation against.
- Your code is **full of host synchronization** that you cannot remove — data-dependent control flow, per-step CPU-side decisions, debugging `.item()` calls woven through the model. Capture will fail, and forcing it is not worth the rewrite unless the model genuinely warrants it.
- You are still **debugging correctness.** Graphs freeze a computation and make its failure modes silent. Get the model correct in eager mode first, then graph it. Never debug numerics through a captured graph.

The one-sentence version, the thing to remember when the details have faded: **CUDA Graphs delete CPU launch overhead by replaying a pre-described, shape-frozen, address-frozen sequence of GPU work — so they are transformative for iterated, launch-bound, small-kernel workloads and pointless for everything else.** Profile first, and profile honestly. If the profile says launch-bound, graphs are likely the highest-leverage change you can make all year. If it does not say launch-bound, then graphs are not your problem, and you should spend your scarce engineering effort somewhere that actually moves the number.

## Further reading

- [NVIDIA CUDA C++ Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) — the authoritative reference for node types, capture semantics, and the update API.
- [NVIDIA Developer Blog — Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) — the original introduction, still the clearest walkthrough of the explicit-construction API.
- [PyTorch — Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — the canonical explanation of `make_graphed_callables`, warmup, and the caching-allocator pool.
- [A survey on LLM acceleration based on KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) — sibling post on the other half of fast LLM decode; the KV cache is the static buffer that makes decode-step capture trivial.
- [Flow matching: a simpler path to generative modeling](/blog/machine-learning/deep-learning/flow-matching) — a sibling deep-learning deep-dive on this blog.
