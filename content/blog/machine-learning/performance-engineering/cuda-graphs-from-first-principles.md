---
title: "CUDA graphs from first principles: capture once, replay a thousand times"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "A launch-bound GPU service pays the CPU-side kernel-launch cost thousands of times a second. A CUDA graph pays it once. This is the intuition, the mechanism, and the math of why capture-and-replay turns a host-bound service into a GPU-bound one."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "cuda-graphs",
    "cuda",
    "profiling",
    "pytorch",
    "latency",
    "throughput",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 42
---

The previous post in this series left you holding a receipt you did not want to pay. We took a small Transformer running decode on an A100, opened the Chrome trace, and found the GPU idle for more than half of every step — not because the math was slow, but because the CPU could not hand the driver work fast enough. Each kernel launch costs a handful of microseconds of pure host-side bookkeeping, and a single decode step fires hundreds of tiny kernels. Multiply five to ten microseconds by seven hundred kernels and you get milliseconds of launch cost per step, sitting on the critical path, while 312 bf16 teraflops of GPU wait for the next instruction. The service ran at 30% GPU utilization and cost real money to keep idle.

That post diagnosed the disease. This post is the cure, and it is almost embarrassingly direct. If the problem is that you pay the launch cost thousands of times per second, the fix is to **pay it once**. A CUDA graph records the entire sequence of kernel launches — the whole directed acyclic graph of GPU operations — into a single replayable object. After that, re-running the step is one driver call that fires the recorded graph, instead of hundreds of driver calls that each drag Python, the dispatcher, and the CUDA runtime along for the ride. The host drops out of the critical path, the GPU runs the kernels back-to-back, and the idle gaps close.

![a diagram showing one capture node branching into four kernel operations that merge back into a single replayable graph handle](/imgs/blogs/cuda-graphs-from-first-principles-1.webp)

This is the conceptual foundation. By the end you will know exactly what a CUDA graph **is** (a frozen DAG of kernels plus their launch parameters, dependencies, and memory addresses), what the capture-instantiate-replay lifecycle means at the raw-CUDA level, and — most importantly — *why* it collapses host overhead from `N × t_launch` per step to roughly one launch per step. You will also understand the price of admission: a graph is frozen, so the shapes, the pointers, and the control flow inside the captured region must be static. That constraint is not a bug; it is the entire reason replay is cheap. The PyTorch API that wraps all of this — `torch.cuda.graph`, `make_graphed_callables`, the graph pool, the static I/O tensors — lives in the sibling post [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch); here we build the mental model the API rests on, so that when you call `torch.cuda.graph(g)` you know precisely what the machine is doing underneath.

## The one-sentence idea: separate deciding-what-to-run from running-it

Every time you write `y = model(x)` in eager PyTorch, two very different kinds of work happen, interleaved so tightly that they look like one thing. There is the **deciding**: Python walks the module tree, the dispatcher picks the right kernel for each operator and dtype, the CUDA runtime allocates argument buffers and computes launch configurations, and finally `cudaLaunchKernel` enqueues the kernel onto a stream. And there is the **running**: the GPU pops the kernel off the stream and executes it on its streaming multiprocessors. The deciding is host work — CPU, Python, driver. The running is device work — SMs, warps, HBM.

In a launch-bound service the deciding is the bottleneck. The GPU finishes each tiny kernel in tens of microseconds and then sits idle waiting for the CPU to finish deciding what comes next. The insight behind CUDA graphs is that **the deciding is identical on every step**. Decode step 1 and decode step 900 run the exact same sequence of kernels with the exact same launch configurations — only the numbers inside the tensors change. So why re-decide seven hundred times a second? Decide once, record the decision, and on every subsequent step just *replay the recording*.

That is the whole idea. A CUDA graph is a recording of the "deciding" — the kernel launches, their arguments, and their dependency structure — captured into a single object that the driver can fire in one call. The "running" still happens every step on real data; only the per-launch host overhead is amortized away. Hold that distinction in your head: **capture freezes the plan, replay executes the plan on live data.**

## What a CUDA graph actually is

Strip away the API and a CUDA graph is a data structure: a directed acyclic graph whose **nodes** are GPU operations and whose **edges** are dependencies. The most common node is a kernel launch, but a node can also be a memory copy (`cudaMemcpyAsync`), a memset, a host callback, or even a nested child graph. An edge from node A to node B means "B may not start until A finishes." The set of nodes with no incoming edges are the roots; they can run as soon as the graph is launched. Everything else waits on its predecessors.

Crucially, each kernel node carries its **launch parameters baked in**: which kernel function, the grid and block dimensions, the shared-memory size, the stream, and — this is the part that trips everyone up later — the actual argument values, including *pointers to the input and output tensors*. When you capture `layernorm(x)`, the node does not store "call layernorm on whatever `x` is"; it stores "call the layernorm kernel with these exact grid/block dims, reading from address `0x7f...a00` and writing to address `0x7f...c80`." The graph is a photograph of one specific execution, pointers and all.

The animated figure below shows the two halves of the model in motion. In the capture phase, the CPU records the four kernels one at a time — paying the full launch cost for each, exactly as in eager mode, because capture *runs* the launches to observe them. In the replay phase, a single `graphLaunch` fires the whole recorded DAG and the GPU executes all four kernels back-to-back with no host gaps between them. Watch how "pay once" on the left becomes "replay cheaply" on the right.

<figure class="blog-anim">
<svg viewBox="0 0 720 250" role="img" aria-label="Two phases of a CUDA graph: in the capture phase the CPU records four kernels one at a time paying a launch cost for each, then in the replay phase a single graph launch fires all four kernels back to back with no gaps" style="width:100%;height:auto;max-width:760px">
<style>
.cg-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cg-name{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.cg-sub{font:600 12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.cg-phase{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.cg-note{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.cg-hi{fill:var(--accent,#6366f1);opacity:.22}
.cg-wipe{fill:var(--accent,#6366f1);opacity:.22;transform-box:fill-box;transform-origin:left center}
.cg-badge{font:700 13px ui-monospace,SFMono-Regular,monospace;fill:var(--accent,#6366f1);text-anchor:middle}
@keyframes cg-fadeA{0%,40%{opacity:1}50%,90%{opacity:0}100%{opacity:1}}
@keyframes cg-fadeB{0%,40%{opacity:0}50%,90%{opacity:1}100%{opacity:0}}
@keyframes cg-sweep{0%,8%{transform:translateX(0)}16%{transform:translateX(170px)}24%{transform:translateX(340px)}32%,100%{transform:translateX(510px)}}
@keyframes cg-fill{0%,50%{transform:scaleX(0)}62%,90%{transform:scaleX(1)}100%{transform:scaleX(0)}}
.cg-A{animation:cg-fadeA 12s ease-in-out infinite}
.cg-B{animation:cg-fadeB 12s ease-in-out infinite}
.cg-sw{animation:cg-sweep 12s steps(1,end) infinite}
.cg-fl{animation:cg-fill 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.cg-A{animation:none;opacity:1}.cg-B{animation:none;opacity:0}.cg-sw{animation:none}.cg-fl{animation:none;transform:scaleX(0)}}
</style>
<text class="cg-phase" x="360" y="26">Capture once, replay a thousand times</text>
<g class="cg-A">
<text class="cg-note" x="360" y="54">Phase 1 · capture: CPU records one kernel at a time, paying a launch cost for each</text>
<rect class="cg-box" x="40"  y="74" width="150" height="70" rx="8"/>
<rect class="cg-box" x="210" y="74" width="150" height="70" rx="8"/>
<rect class="cg-box" x="380" y="74" width="150" height="70" rx="8"/>
<rect class="cg-box" x="550" y="74" width="130" height="70" rx="8"/>
<rect class="cg-hi cg-sw" x="40" y="74" width="150" height="70" rx="8"/>
<text class="cg-name" x="115" y="104">k1 matmul</text>
<text class="cg-sub"  x="115" y="126">+ launch 8µs</text>
<text class="cg-name" x="285" y="104">k2 bias</text>
<text class="cg-sub"  x="285" y="126">+ launch 8µs</text>
<text class="cg-name" x="455" y="104">k3 norm</text>
<text class="cg-sub"  x="455" y="126">+ launch 8µs</text>
<text class="cg-name" x="615" y="104">k4 gelu</text>
<text class="cg-sub"  x="615" y="126">+ launch 8µs</text>
<text class="cg-note" x="360" y="182">4 launches · host pays each · GPU waits in the gaps</text>
</g>
<g class="cg-B">
<text class="cg-note" x="360" y="54">Phase 2 · replay: one graphLaunch fires the whole DAG back to back</text>
<rect class="cg-box" x="40"  y="74" width="150" height="70" rx="8"/>
<rect class="cg-box" x="210" y="74" width="150" height="70" rx="8"/>
<rect class="cg-box" x="380" y="74" width="150" height="70" rx="8"/>
<rect class="cg-box" x="550" y="74" width="130" height="70" rx="8"/>
<rect class="cg-fl" x="40" y="74" width="640" height="70" rx="8"/>
<text class="cg-name" x="115" y="112">k1</text>
<text class="cg-name" x="285" y="112">k2</text>
<text class="cg-name" x="455" y="112">k3</text>
<text class="cg-name" x="615" y="112">k4</text>
<text class="cg-badge" x="360" y="166">graphLaunch × 1</text>
<text class="cg-note" x="360" y="182">1 launch · GPU runs all four back to back · no gaps</text>
</g>
</svg>
<figcaption>Phase 1 pays the launch cost once, recording each kernel into the graph; phase 2 replays the whole recorded DAG with a single launch, so the GPU runs the kernels back to back with no host gaps.</figcaption>
</figure>

One more thing before we go deeper: a graph is not a compiler. It does not fuse your kernels, rewrite your math, or make any individual kernel faster. That is the job of `torch.compile` and Inductor, covered later in this series and cross-linked below. A graph is a *dispatch optimization* — it makes launching the kernels nearly free. The two compose beautifully (compile fuses, then graphs the fused result — that is what `mode="reduce-overhead"` does), but keep them separate in your mind. Fusion attacks redundant work and memory traffic; graphs attack launch overhead. They kill different wastes.

## The capture, instantiate, replay lifecycle

At the raw-CUDA level, turning a stream of kernel launches into a graph has three distinct stages, and understanding each one demystifies everything the PyTorch API does for you. The lifecycle, in order, is: **warm up → capture → instantiate → replay**. The figure lays them out on a timeline; then we walk each stage.

![a timeline showing warmup then begin capture then run kernels then end capture then instantiate then replay repeated many times](/imgs/blogs/cuda-graphs-from-first-principles-2.webp)

**Warm up (before capture).** This stage is not part of the CUDA graph API, but skipping it is the single most common way to get a broken or slow graph. When a kernel runs for the first time, a lot of one-time work happens: cuDNN and cuBLAS autotune to pick the best algorithm for your shapes, the caching allocator carves out fresh memory blocks, lazy module state initializes, and the JIT may compile a kernel. If any of that happens *during* capture, you either record the autotuning launches into your graph (garbage) or capture pointers that will not be valid on replay. So you run the exact workload a few times first, on a side stream, to force all the one-time work to complete. Then you capture a clean, steady-state sequence.

**Begin capture.** You call `cudaStreamBeginCapture(stream)`. This flips the stream into *capture mode*. From this point, every operation you enqueue on that stream is **recorded, not executed for keeps**. The runtime intercepts each `cudaLaunchKernel`, each `cudaMemcpyAsync`, and instead of just submitting it, it adds a node to the graph being built and wires up the dependency edges based on the stream and event ordering. This is the subtle part worth saying twice: in capture mode the launches still *appear* to run — the CPU still does the per-launch work, which is why warmup matters — but the driver is transcribing them into a graph rather than letting the GPU treat them as the final word.

**Run your kernels.** Between begin and end, you execute your normal forward pass — the same sequence of ops you want to replay. The runtime records each one. Dependencies are inferred from stream semantics: two ops on the same stream get a dependency edge (serialized), and cross-stream dependencies expressed via CUDA events become edges too. The result is a DAG that faithfully encodes the parallel structure of your computation.

**End capture.** You call `cudaStreamEndCapture(stream, &graph)`, which stops recording and hands you a `cudaGraph_t` — a template describing the topology. Note the word *template*: the `cudaGraph_t` is a description, not yet an executable thing.

**Instantiate.** You call `cudaGraphInstantiate(&graphExec, graph, ...)`, which turns the template into an **executable graph** (`cudaGraphExec_t`). This is where the driver does the expensive one-time preparation: it validates the DAG, computes an execution schedule, and lays out the launch metadata so that firing the graph later is as cheap as possible. Instantiation is not free — it can cost as much as a normal launch or more — but you do it **once**, so its cost is amortized across every replay.

**Replay.** You call `cudaGraphLaunch(graphExec, stream)`. This is the payoff: a single driver call that submits the entire recorded DAG to the GPU. No Python, no per-op dispatch, no per-kernel `cudaLaunchKernel`. The driver already knows every kernel, every argument, every dependency; it just fires the whole thing. You can call `cudaGraphLaunch` as many times as you like — a thousand times, a million times — and each call is that same single cheap submission.

Here is the raw-CUDA skeleton, first as the mental shape and then as real API calls, so the correspondence is exact:

```cpp
// --- Stage 0: warm up so autotuning + allocation happen BEFORE capture ---
for (int i = 0; i < 3; ++i) {
    run_step(inputBuf, outputBuf, side_stream);   // same shapes as real work
}
cudaStreamSynchronize(side_stream);

// --- Stage 1: capture (record, do not commit) ---
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
run_step(inputBuf, outputBuf, stream);            // enqueues k1..k4; recorded as nodes
cudaStreamEndCapture(stream, &graph);             // graph = a DAG template

// --- Stage 2: instantiate (compile the template once) ---
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, /*flags=*/0);

// --- Stage 3: replay (fire the whole DAG with ONE driver call, N times) ---
for (int step = 0; step < N; ++step) {
    fill_inputs(inputBuf);                        // write NEW data into the SAME buffer
    cudaGraphLaunch(graphExec, stream);           // 1 launch replays all kernels
}
cudaStreamSynchronize(stream);
```

Read the replay loop carefully, because it contains the whole contract. `fill_inputs(inputBuf)` writes fresh data into `inputBuf` — the *same* memory address the graph recorded. The graph does not care what numbers live at that address; it only cares that the address is the one it recorded. Then `cudaGraphLaunch` fires all the kernels, which read from that address, do the math on whatever is there now, and write the result to the recorded output address. **The plan is frozen; the data is live.** That is why a graph can replay a thousand times and produce a thousand different results.

The mapping from these raw calls to what you will actually type in PyTorch is one-to-one. `torch.cuda.graph(g)` is a context manager that calls `cudaStreamBeginCapture` on entry and `cudaStreamEndCapture` + `cudaGraphInstantiate` on exit; `g.replay()` is `cudaGraphLaunch`; the "static input tensor" you are told to write into is exactly `inputBuf` above. The sibling post [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) walks that API in full — the graph pool, `make_graphed_callables`, wiring it into a real forward pass — so I will not duplicate it here. What matters now is that when you read that API, you can see the raw calls behind it.

### Capture records; it does not execute for keeps

The stage that confuses people most is capture, because during capture your kernels *look* like they are running. You call `model(x)` inside the `with torch.cuda.graph(g):` block and it returns; no exception, no obvious difference from an eager call. So it is worth being precise about what is and is not happening.

During capture, the host still does the full per-launch work — Python walks the module tree, the dispatcher resolves each operator, the runtime marshals arguments, and it issues each `cudaLaunchKernel`. That is why capture is *not* faster than an eager step and why warmup must precede it: capture pays the launch cost exactly once, as an eager step would. The difference is what the *driver* does with each submitted launch. Instead of forwarding it to the GPU as a committed piece of work, the driver in capture mode intercepts it and appends a node to the graph under construction, recording the kernel handle, the launch configuration, and the argument pointers. The GPU does not necessarily run the recorded kernels to produce a usable result during capture — the point of capture is to *observe and transcribe* the launch sequence, not to compute an answer you will use.

The cleanest one-line summary: **eager mode says "run this now," capture mode says "write down that we would run this."** After `cudaStreamEndCapture`, you hold a description of what would have run — the DAG — and nothing has been committed to the GPU as final work. Instantiate then compiles that description into an executable, and replay is the first time the recorded sequence actually runs as committed GPU work, on whatever data currently lives at the recorded addresses. Internally, this is why a stray `.item()` during capture is fatal: reading a value back to the CPU requires the GPU to actually *finish and return* a result mid-capture, but capture is transcribing, not committing, so there is nothing finished to read. The synchronization has no valid meaning in capture mode, and the runtime rejects it.

### Capture modes: how strict the driver is about stray work

`cudaStreamBeginCapture` takes a capture-mode argument that controls how paranoid the driver is about operations *outside* the captured stream leaking in. Three modes exist, in decreasing strictness:

- **Global** — the strictest. Any potentially-unsafe CUDA call anywhere in the process during capture (for example, a synchronizing call on a different stream) invalidates the capture. Use it when you want the driver to catch every possible interference.
- **Thread-local** — the common default in framework integrations. Only unsafe calls on the *capturing thread* invalidate the capture, so other threads doing unrelated CUDA work do not spoil your graph. This is what the PyTorch path typically uses, and it is the mode in the skeleton above.
- **Relaxed** — the most permissive. You take responsibility for not doing anything unsafe; the driver stops policing. Reach for it only when you know exactly what the rest of the process is doing.

You rarely set this by hand in PyTorch — the framework picks a sensible mode — but knowing the knob exists explains a whole class of "capture failed because some other thread did X" errors: the driver's capture-mode policing caught cross-stream or cross-thread work that would have made the recorded graph unsound. It is the same frozen-DAG contract enforced from a different angle: the graph can only faithfully record a *clean, isolated* sequence, and the capture mode decides how hard the driver works to guarantee that isolation.

| Raw CUDA call | What it means | PyTorch equivalent |
| --- | --- | --- |
| `cudaStreamBeginCapture` | start recording the stream | entering `with torch.cuda.graph(g):` |
| run your kernels | ops recorded as graph nodes | the body of the `with` block |
| `cudaStreamEndCapture` | stop recording, get a template | exiting the `with` block |
| `cudaGraphInstantiate` | compile the template once | (also on `with` exit) |
| `cudaGraphLaunch` | fire the whole DAG, one call | `g.replay()` |

## Why it works: the host-cost collapse

Now the mechanism — the part that turns "sounds nice" into "here is the number." Let me build the cost model from the ground up so the saving is provable, not asserted.

Define `t_launch` as the CPU-side cost to launch one kernel: the Python dispatch, the operator resolution, the CUDA runtime's argument marshalling, and the `cudaLaunchKernel` call. On a modern host this is roughly 5 to 10 microseconds per kernel in an eager PyTorch pipeline — call it `t_launch ≈ 8` µs as a working figure (NVIDIA's own CUDA-graphs material cites single-digit-microsecond launch overhead, and PyTorch's dispatcher adds several more). Let `N` be the number of kernels your service launches per step, and `t_gpu,i` the GPU execution time of kernel `i`.

In **eager mode**, the host must issue all `N` launches every step. The host-side cost per step is:

$$
t_\text{host}^\text{eager} = N \cdot t_\text{launch}
$$

The GPU can only run kernel `i` once the host has launched it. If the host cannot stay ahead — that is, if `N · t_launch` grows faster than the GPU can drain the queue — the GPU stalls between kernels, waiting for the next launch. The step time is then dominated not by the GPU work but by the host's launch cadence:

$$
t_\text{step}^\text{eager} \approx \max\!\Big( \sum_i t_{\text{gpu},i},\; N \cdot t_\text{launch} \Big)
$$

When you are launch-bound, the second term wins. This is precisely the signature from the launch-overhead post: the GPU timeline is a picket fence of short kernels separated by idle gaps, and the total step time tracks the number of kernels, not the amount of math.

Now **graph mode**. Capture pays the launch cost once, at setup, so it is amortized to zero across a long-running service. On every step, replay issues *one* driver call whose host cost is `t_graph`, the cost of `cudaGraphLaunch` — a fixed, tiny amount, roughly one launch's worth of overhead regardless of how many kernels the graph contains. So:

$$
t_\text{host}^\text{graph} = t_\text{graph} \approx t_\text{launch}
$$

The host cost per step went from `N · t_launch` to about `t_launch`. The reduction factor is `N` — the number of kernels in the step. And with the host out of the way, the GPU is now free to run the recorded kernels back-to-back at whatever pace it can sustain, so the step time falls to the GPU floor:

$$
t_\text{step}^\text{graph} \approx \sum_i t_{\text{gpu},i}
$$

The before-and-after figure makes the collapse concrete on the running example.

![a before and after comparison showing eager mode with 700 launches and 12 ms host time and 60 percent idle versus graphed mode with one launch and near zero host time and back to back kernels](/imgs/blogs/cuda-graphs-from-first-principles-3.webp)

#### Worked example: the host-time collapse on an A100

Take the decode step from the previous post: a small Transformer at batch size 1 launching `N = 700` kernels per step, on an A100 80GB. With `t_launch ≈ 8` µs:

- Eager host cost per step: `700 × 8 µs = 5,600 µs = 5.6 ms`. The pure GPU work in that step — the actual matmuls, norms, and elementwise ops — sums to only about `2.0 ms`. Because `5.6 ms > 2.0 ms`, the step is launch-bound: it takes about `5.6 ms`, and the GPU is busy only `2.0 / 5.6 ≈ 36%` of the time. That matches the ~30% utilization we measured.
- Graph host cost per step: one replay, `t_graph ≈ 8 µs = 0.008 ms`. Now `0.008 ms ≪ 2.0 ms`, so the step is GPU-bound and takes about `2.0 ms`. Utilization jumps toward `2.0 / 2.05 ≈ 97%` (the small remainder is the replay call and the odd bubble).

Step time fell from `5.6 ms` to about `2.0 ms` — a `2.8×` speedup — and it came entirely from deleting host overhead, not from making a single kernel faster. Throughput at batch 1 rises from `1000 / 5.6 ≈ 178` steps/s to `1000 / 2.0 = 500` steps/s. That is the shape of a launch-bound win: the speedup is roughly `min(N · t_launch, Σt_gpu) / Σt_gpu`, capped by how launch-bound you were to begin with.

The ceiling here is worth stating plainly, because it is Amdahl's law wearing a CUDA hat. Graphs remove host launch cost; they cannot remove GPU work. If your step were already GPU-bound — say `Σt_gpu = 10 ms` and `N · t_launch = 1 ms` — then eager step time is `max(10, 1) = 10 ms`, graphed step time is still `10 ms`, and you have saved essentially nothing. The maximum speedup from graphing is `t_step^eager / Σt_gpu`, which only exceeds ~1.1× when host cost is a large fraction of the step. **Graphs are a launch-overhead fix, and only a launch-overhead fix.** We will make this precise in the "where the win is biggest" section.

### The setup cost, and when it pays for itself

Capture and instantiate are not free. Capture pays one full step's worth of launch cost, and instantiate can cost as much as a kernel launch or several — the driver validates the DAG and lays out its execution schedule. Call the combined one-time setup cost `t_setup`. You only come out ahead once the accumulated per-step savings exceed it. If graphing saves `Δ = t_step^eager − t_step^graph` per step, the break-even step count is:

$$
S_\text{break-even} = \frac{t_\text{setup}}{\Delta}
$$

#### Worked example: how many steps before graphing pays off

On the A100 decode example, `Δ = 5.61 − 2.03 = 3.58` ms saved per step. Suppose capture plus instantiate costs `t_setup ≈ 20` ms (one eager step of ~5.6 ms to capture, plus ~14 ms to instantiate — order-of-magnitude; measure yours). Then `S_break-even = 20 / 3.58 ≈ 6` steps. After roughly six replays you are ahead, and every replay after that is pure profit. For a service that replays the same graph millions of times a day, `t_setup` is a rounding error paid once at startup — which is exactly why graphs are a serving and long-running-training optimization, not something you would bother capturing for a handful of iterations. If you only need to run a region five times, do not graph it; the setup will not amortize.

This also explains why you capture at **process startup**, warm and off the request path, not lazily on the first request. The first replay a user sees should be a cheap `cudaGraphLaunch`, not a capture-plus-instantiate stall that spikes that request's tail latency. Amortize the setup during warmup, before you accept traffic.

### A bonus: the DAG can overlap what a single stream serializes

There is a second, subtler win hiding in the word *graph*. When you launch kernels one at a time on a single stream in eager mode, they run strictly in issue order — the stream is a queue. But a captured graph is a *DAG*, and its edges encode only the real dependencies. If two branches of your computation are genuinely independent — say two attention heads' projections, or a weight update on one tensor and a metric computation on another — the graph knows they do not depend on each other, and the driver is free to schedule them concurrently on the GPU where resources allow. In eager single-stream code that concurrency is left on the table because everything serializes through one queue.

So a graph can, in the right topology, deliver a little extra speedup beyond the pure launch-overhead saving, by exposing parallelism the frozen dependency structure makes explicit. Do not over-count this — most of the win in a launch-bound decode loop is still the `N · t_launch → t_graph` collapse, and the kernels there are usually a dependent chain with little to overlap. But it is real, and it is why the data structure is a *graph* and not merely a flat *recording*: the dependency edges carry scheduling freedom, not just ordering. The deeper treatment of squeezing concurrency out of independent work lives in this series' [overlapping streams and concurrency](/blog/machine-learning/performance-engineering/overlapping-streams-and-concurrency) post; graphs give you some of it automatically as a side effect of capturing the true DAG.

## What a graph freezes, and why static addresses are mandatory

We keep saying the graph is "frozen." Let us be exact about *what* is frozen, because that list is the source of every constraint and every gotcha you will hit. A captured graph freezes four things together.

![a layered stack showing that a graph freezes kernels then launch parameters then dependencies then memory addresses down to one executable handle](/imgs/blogs/cuda-graphs-from-first-principles-4.webp)

1. **The kernels and their order** — which kernel functions run, and the DAG of dependencies between them. Replay runs exactly this set in exactly this partial order.
2. **The launch parameters** — grid dimensions, block dimensions, shared-memory size, stream. These were computed from the shapes at capture time and are now constants in the graph.
3. **The dependencies** — the edges. Replay respects them, allowing independent branches to overlap and forcing dependent nodes to serialize, just as capture recorded.
4. **The memory addresses** — and this is the one that owns most of the debugging. Every kernel node stores the literal pointers it reads from and writes to. Not "the input tensor" — the *address* `0x7f2a...`. On replay, the kernels read and write those same addresses, no questions asked.

Point 4 is why the number-one rule of CUDA graphs is **static memory addresses**. The graph is a photograph that includes the pointers. If, between capture and replay, your input tensor moves to a different address — because you allocated a fresh tensor, because the caching allocator handed you a different block, because you reshaped in a way that reallocated — the graph will happily replay reading from the *old* address, which now contains stale data or nothing at all. The output is garbage, and there is no error, because from the GPU's point of view nothing is wrong: it dereferenced exactly the pointer it was told to.

The correct pattern, which the raw-CUDA skeleton above already showed, is to allocate your input and output buffers **once**, before capture, and on every step **copy new data into those same buffers in place** rather than creating new tensors. `inputBuf.copy_(new_data)` keeps the address stable; `inputBuf = new_data` swaps the address and breaks the graph. This is exactly why PyTorch's graph API insists you hand it "static input tensors" and write into them with `copy_`.

The same logic explains the other hard requirements. **Static shapes**, because the launch parameters (point 2) were computed from the shapes at capture; a different shape needs different grid dimensions, which the frozen graph does not have. **No data-dependent control flow inside the region**, because a captured graph has no `if` — it is a fixed DAG, so "run kernel A if the logit is positive, else kernel B" cannot be recorded; both the branch condition and the choice happen on the host, which is not in the replay path. **No disallowed synchronization**, because a sync that pulls a value back to the CPU mid-region (a `.item()`, a `.cpu()`, a `print(tensor)`) both breaks capture and implies host-side control that the graph cannot contain.

| What a graph freezes | What that forces on you | Why |
| --- | --- | --- |
| Memory addresses (pointers) | reuse buffers, `copy_` in place | replay dereferences the recorded address |
| Launch params (grid/block) | shapes must be static | grid dims were computed from capture shapes |
| The DAG topology | no data-dependent branching | a frozen DAG has no `if` on the host |
| The kernel set | no `.item()` / `.cpu()` in region | host round-trips are not in the replay path |

None of this is a defect. It is the *reason* replay is cheap: because the graph committed to every pointer, dimension, and edge up front, launching it requires zero decisions at runtime. The rigidity buys the speed. When your workload naturally has this rigidity — a fixed-shape inference server, a decode loop with a fixed context layout — graphs are nearly free money. When it does not, you have to *manufacture* the rigidity (bucket your shapes, pad to fixed sizes, hoist control flow out of the region), and that engineering is the subject of the gotchas post.

## Can I graph this? The hard requirements as a decision

Before you reach for a graph, run the region through four gates. If it passes all four, capture will succeed and replay will be correct. If it fails any, you either fix the region to satisfy the gate or you do not graph it. The decision tree below is the checklist; the forward-linked [CUDA graphs gotchas and debugging](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) post is what to do when a gate fails and you still want the win.

![a decision tree splitting the graphability question into static gates on shapes and memory and dynamic risk gates on control flow and synchronization](/imgs/blogs/cuda-graphs-from-first-principles-5.webp)

The two static gates are the ones you control by construction:

- **Static shapes.** Every tensor in the region must have the same shape on every replay. Decode at a fixed context length, inference at a fixed batch size, or a padded/bucketed shape all qualify. Variable sequence lengths per request do not — until you bucket them, which is the serving trick covered later in the track.
- **Static addresses.** Inputs and outputs live in buffers allocated once and reused. In PyTorch this also means the region must not trigger the caching allocator to hand back a different block, which is why you warm up first (to let the allocator settle) and why graph capture uses a dedicated memory pool so that replay's allocations land at the same addresses every time.

The two dynamic-risk gates are the ones that quietly break correctness:

- **No CPU-side control flow.** No branch inside the region whose condition is read from a GPU tensor. No early-exit based on `if loss.item() < threshold`. No Python loop whose trip count depends on data. Structural control flow that is *fixed* at capture time is fine (a `for` over the 32 transformer layers is fine — it always runs 32 times); *data-dependent* control flow is not.
- **No illegal synchronization.** No `.item()`, `.cpu()`, `torch.nonzero` with dynamic output, or any op that forces a device-to-host round-trip inside the captured region. These both break the capture (you cannot synchronize a stream that is being captured) and signal host-side logic the graph cannot hold.

#### Worked example: three ways a decode loop fails the gates

Suppose you try to graph a single decode step and it misbehaves. Reason through which gate it tripped:

- **Symptom: the output is subtly wrong on replay, no error.** You are probably failing the static-address gate. Somewhere in the region you allocated a fresh tensor — maybe `logits = model.head(h)` returns a new tensor each call, or you did `x = torch.cat([...])` that reallocated. Capture recorded the address of the *first* allocation; on replay the allocator may hand you a different block for the live compute while the graph still reads the recorded one. Fix: pre-allocate the buffer and write into it with `copy_`, or route the region through PyTorch's graph pool so allocations are deterministic.
- **Symptom: capture raises an error about synchronization or an operation not permitted during capture.** You are failing the no-illegal-sync gate. There is a `.item()`, a `.cpu()`, a `print(tensor)`, or an implicit sync (a `.numpy()`, a boolean check on a tensor) inside the region. Fix: hoist that host round-trip out of the captured region — do the sampling/argmax that needs a CPU value *after* replay, not inside it.
- **Symptom: the graph is correct at context length 128 but garbage at 200.** You are failing the static-shape gate. The KV cache or the attention mask changed shape, so the recorded grid dimensions no longer match the work. Fix: capture at a fixed maximum context and mask, or bucket context lengths and keep one graph per bucket.

Notice that only the second symptom gives you an error. The first and third fail *silently* — the GPU does exactly what the frozen graph says, which is the wrong thing for the new data. This is why "correct output" is a required part of validating a graph, not an assumed one, and why the gotchas post spends most of its length on silent-corruption debugging.

## Where the win is biggest, and where it is nothing

The cost model already told us the punchline: the speedup is bounded by how launch-bound you were. Let us turn that into a decision you can make at a glance. The matrix scores common workloads by whether they are launch-bound and how much graphing buys.

![a matrix rating workloads such as decode loops tiny elementwise chains cnn inference big batch gemm and long sequence attention by whether they are launch bound and how much a graph helps](/imgs/blogs/cuda-graphs-from-first-principles-6.webp)

The pattern is monotone in one variable: **kernels per step relative to GPU work per step.** A workload wins from graphing exactly when it launches many small kernels — because then `N · t_launch` is large relative to `Σt_gpu`, and deleting the host cost uncovers a lot of idle GPU.

- **Decode at batch 1** is the poster child. Hundreds of tiny kernels, each finishing in tens of microseconds, so the host cannot keep up. Published results and NVIDIA's own guidance put graph speedups here at roughly 1.5× to 3× end-to-end for launch-bound decode, and the effect grows as batch size shrinks (fewer FLOPs per kernel, so launch cost dominates more).
- **Tiny elementwise chains** — a long sequence of adds, multiplies, activations, each a separate kernel — are similar: almost all launch, almost no math per kernel.
- **CNN inference** is in the middle. Convolutions are chunkier than decode kernels, so a smaller fraction of the step is launch overhead, but a ResNet still fires a hundred-plus kernels per forward pass and typically sees a real if smaller graph win.
- **A single big-batch GEMM** is the anti-example. One large matmul that runs for milliseconds has a launch cost of a few microseconds — utterly negligible. Graphing it saves nothing, because there was nothing to save. The same goes for a large fused attention kernel processing a long sequence: the GPU is already the bottleneck.

This is the same message as the Amdahl ceiling, restated as a rule you can act on: **graph the launch-bound regions; leave the compute-bound ones alone.** If `nvidia-smi` shows high utilization and the Chrome trace shows fat kernels back-to-back with no gaps, a graph will do nothing for you — and it will *cost* you flexibility (static shapes, static memory) for that nothing. The next section quantifies the win where it does exist.

## Measuring the win honestly

A CUDA graph's whole value proposition is a before-and-after number, so you have to measure it in a way you can trust. The traps here are the same ones that fool everyone timing GPU code, plus one specific to graphs.

First, the universal rules. The GPU runs asynchronously, so wall-clock time around a Python call measures when the *launch* returned, not when the *kernel* finished. Always `torch.cuda.synchronize()` before you read the clock, or use CUDA events (`torch.cuda.Event(enable_timing=True)`) which timestamp on the device. Always warm up first — the first few iterations pay autotuning and allocation costs that steady state does not — and discard them. Lock the clocks (`nvidia-smi -lgc`) if you can, so thermal throttling does not masquerade as a regression. Measure steady state over many iterations and report a distribution, not one sample. All of this is the subject of the [reproducible-benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) post in Track A; here is the graph-specific harness:

```python
import torch

# static I/O buffers, allocated ONCE (addresses the graph will freeze)
static_in  = torch.randn(1, 512, 768, device="cuda")
static_out = torch.empty(1, 512, 768, device="cuda")

def step():
    # your fixed-shape region; writes into static_out in place
    static_out.copy_(model(static_in))

# --- warm up on a side stream so autotuning/alloc finish before capture ---
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        step()
torch.cuda.current_stream().wait_stream(s)

# --- capture + instantiate (once) ---
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    step()                      # recorded, not committed

def timed(fn, iters=200):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters      # ms per step

eager_ms  = timed(step)                          # eager path
graph_ms  = timed(lambda: g.replay())            # replay path: NEW data would go in static_in first
print(f"eager {eager_ms:.3f} ms   graph {graph_ms:.3f} ms   speedup {eager_ms/graph_ms:.2f}x")
```

Running that on the launch-bound decode example on an A100 prints something like:

```console
eager 5.612 ms   graph 2.031 ms   speedup 2.76x
```

The graph-specific trap: **measure the replay path the way production runs it**, which means write fresh data into `static_in` before each `g.replay()`. If you skip the input copy in your benchmark, you are timing replay of stale data and slightly understating the real per-step cost (the copy is cheap, but be honest about it). Conversely, do not accidentally include the one-time capture and instantiate cost in your per-step average — those are setup, amortized over the life of the service, not part of steady-state step time.

The other honest check is *correctness*, which timing cannot see. After you graph a region, assert that `g.replay()` on a given input produces the same tensor (within floating-point tolerance) as the eager path on that same input. A graph that is 3× faster and subtly wrong is worse than no graph. Because the failure mode is silent (stale pointers, wrong shape), a numerical equality check against the eager reference is the only thing standing between you and a service that quietly serves garbage.

## Named-hardware before and after

Here is the full picture on two very different GPUs, running the same fixed-shape small-Transformer decode step at batch size 1. The A100 is our compute-rich datacenter part; the L4 is a cheaper inference card with far less compute and bandwidth. Both are launch-bound at batch 1 for the same reason — the host cannot keep up with the tiny kernels — so both benefit, and the numbers below are the kind you would read straight off the harness above.

**A100 80GB SXM, small Transformer decode, batch 1, `N ≈ 700` kernels/step:**

| Metric | Eager | Graphed | Change |
| --- | --- | --- | --- |
| Host launch time / step | 5.6 ms | 0.02 ms | −99.6% |
| GPU idle per step | ~60% | ~8% | gaps closed |
| Step time (p50) | 5.61 ms | 2.03 ms | 2.76× faster |
| Throughput | 178 steps/s | 492 steps/s | 2.76× |
| GPU utilization | ~34% | ~92% | host removed |
| Kernels launched / step (host) | 700 | 1 | 700→1 |

**L4, same fixed-shape decode, batch 1:**

| Metric | Eager | Graphed | Change |
| --- | --- | --- | --- |
| Host launch time / step | 5.6 ms | 0.02 ms | −99.6% |
| Step time (p50) | 8.9 ms | 5.7 ms | 1.56× faster |
| Throughput | 112 steps/s | 175 steps/s | 1.56× |
| GPU utilization | ~46% | ~78% | host removed |

The comparison teaches the mechanism better than either table alone. The **host cost is identical** on both cards — `700 × 8 µs` of launch overhead does not care which GPU it is stalling, because it is CPU work. But the L4's kernels take longer (less compute, less bandwidth), so on the L4 the GPU work per step is larger relative to the fixed host cost — the step was "less launch-bound" to begin with. Graphing still removes the full 5.6 ms of host time, but because the L4's GPU floor (`Σt_gpu ≈ 5.7 ms`) is higher, the *ratio* improvement is smaller: 1.56× instead of 2.76×. Same fix, same absolute host saving, different speedup — exactly what `speedup = t_step^eager / Σt_gpu` predicts. This is why "how much will graphs help me?" has no single answer: it depends on your GPU floor.

## A problem-solving narrative: from 30% to 90% util

Let me walk the whole loop once, end to end, the way it actually goes when you are paged for it — profile, hypothesize, fix, re-measure, stress-test.

**The symptom.** An inference service serving a small Transformer at batch 1 runs at 30% GPU utilization on an A100 and costs about \$40 per GPU-day to keep barely busy. Throughput is capped around 180 requests per second per GPU and will not budge no matter how many requests you throw at it. Adding replicas scales linearly but expensively; something is wrong per-GPU.

**Read the profile.** You attach `torch.profiler` and export a Chrome trace ([reading the trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) is its own post). The GPU-kernel lane is a picket fence: hundreds of kernels, each 15 to 60 µs long, separated by idle gaps of comparable width. The CPU lane is packed solid with `cudaLaunchKernel` and dispatcher frames. `key_averages()` shows the top "cost" is not any single kernel but the sheer *count* — seven hundred launches per step. This is the launch-bound signature the previous post taught you to recognize: the GPU is idle not because kernels are slow but because it is starved for launches.

**Hypothesize.** If the step is launch-bound at `N · t_launch = 700 × 8 µs = 5.6 ms` while the GPU work is only ~2.0 ms, then removing the per-launch host cost should drop step time to ~2.0 ms and lift utilization toward 90%. The shapes are fixed (batch 1, fixed context), there is no data-dependent branching in the forward pass, and sampling happens after the forward pass returns — so the region passes all four graphability gates. A CUDA graph is the indicated fix.

**Apply the fix.** Allocate static input/output buffers, warm up on a side stream, capture the forward pass into a graph, and replace the eager forward with `g.replay()` after copying the request's data into the static input. (The [PyTorch API post](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) is the full recipe; `mode="reduce-overhead"` in `torch.compile` is the one-line version that does capture for you, covered in Track D.)

**Re-measure.** Step time falls from 5.61 ms to 2.03 ms, utilization rises from ~34% to ~92%, throughput goes from 178 to 492 requests/s per GPU. The picket fence in the trace is gone — the kernels now run shoulder to shoulder — and the CPU lane, formerly wall-to-wall launches, is nearly empty during the step. You just tripled the throughput of every GPU in the fleet without buying a single new one, which turns \$40 of mostly-idle GPU-day into \$40 of mostly-busy GPU-day.

**Stress-test it.** This is where junior fixes fall over and senior ones earn their keep:

- **Batch 64 instead of batch 1.** Now each kernel does 64× the work, so `Σt_gpu` grows while `N · t_launch` stays flat. The step becomes compute-bound; the eager utilization is already ~85% and the graph speedup shrinks to ~1.1×. Still worth it, barely, but the win is small — exactly as the matrix predicted. The graph did not get worse; the workload got less launch-bound.
- **Shapes vary per request.** If sequence length changes per request, a single graph captured at one length is wrong for the others. You bucket: capture one graph per shape bucket (128, 256, 512), route each request to its bucket's graph, and eat a little padding waste. That is the serving-loop post's whole topic; the point here is that variable shapes do not kill graphs, they just require you to manufacture static shapes.
- **On an L4 instead of an A100.** The GPU floor is higher, so the speedup drops to ~1.5× as we saw. Still a real win, but you would set expectations differently.
- **A `.item()` sneaks into the region.** Someone adds an early-exit `if logits.max().item() > τ: break` inside the captured forward. Capture now throws — you cannot sync a stream mid-capture. The fix is to move the host round-trip out of the region and do the check after replay. This is a feature: the error caught a real violation of the frozen-DAG contract before it could corrupt anything.

## Case studies and published numbers

A few grounded data points, so the "2 to 3×" is not just my example talking.

**PyTorch `reduce-overhead` mode.** `torch.compile(model, mode="reduce-overhead")` composes Inductor fusion with CUDA graphs — it captures the compiled kernels into a graph so both the redundant-work waste and the launch-overhead waste are attacked at once. The PyTorch team's own writeups on this mode report meaningful end-to-end speedups on inference workloads that are launch-bound, and the mode exists precisely because so many small-batch inference paths are dominated by launch overhead. The composition is the natural end state: fuse first (fewer, larger kernels), then graph the result (near-zero launch cost for what remains). This series' Track D post [compile plus CUDA graphs](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead) covers exactly how they compose and where the memory and dynamic-shape pitfalls hide.

**NVIDIA's CUDA Graphs guidance.** NVIDIA introduced graphs specifically to attack the "launch a huge number of short-running kernels" pattern, and their developer material demonstrates that when per-kernel launch overhead is a significant fraction of kernel runtime, replacing per-kernel launches with a single graph launch removes that overhead almost entirely. Their canonical example — a loop of many tiny kernels — is the clean demonstration of `N · t_launch → t_graph`, and it is the direct ancestor of every decode-loop speedup you will see in an LLM serving stack. The model-serving series covers the production integration in [kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile).

**LLM decode in production serving.** Inference engines that serve autoregressive decode at low batch sizes lean heavily on CUDA graphs because decode is the launch-bound case par excellence: one token at a time, hundreds of tiny kernels per token, the GPU starved for launches. Capturing the decode step per batch-size bucket and replaying it is now standard practice in high-throughput serving stacks, and it is a large part of why those stacks hit high GPU utilization at small batch. The exact numbers vary by model and stack, but the direction is always the same and always for the same reason: decode is host-bound until you graph it. The serving-loop details — how to capture one graph per batch-size bucket, how it coexists with continuous batching and a paged KV cache, and how a real service goes host-bound to GPU-bound — are the subject of this track's [CUDA graphs in a serving loop](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) post.

**Vision inference and the middle ground.** A ResNet or a detection backbone at inference time is the CNN case from the matrix: a hundred-plus convolution and normalization kernels per forward pass, chunkier than decode kernels but still numerous. Graphing such a model typically buys a modest but real speedup — enough to matter at fleet scale — and it composes with the channels-last memory-format win covered in Track E, because the two attack different wastes (layout/bandwidth versus launch overhead) and stack rather than overlap. The lesson that recurs across all three case studies is the same one the cost model predicts: the graph win is proportional to how much of your step was host-side launch, and nothing more.

## When to reach for this, and when not to

A CUDA graph is a trade: you spend flexibility (static shapes, static memory, no in-region control flow) to buy the elimination of launch overhead. Make the trade when the overhead is real and the flexibility is cheap to give up.

**Reach for a graph when:**

- Your profile shows the launch-bound signature — many short kernels, GPU idle in the gaps, CPU lane full of launches, step time tracking kernel *count* not kernel *cost*.
- The region has naturally static shapes and no data-dependent control flow — fixed-shape inference, a decode step at a fixed context, a fixed training step. The gates pass without contortion.
- You run the same region many, many times, so the one-time capture/instantiate cost amortizes to nothing.

**Do not reach for a graph when:**

- You are **compute-bound** — high utilization, fat kernels back-to-back, no gaps. There is no launch overhead to remove, and you would pay the rigidity for nothing. Profile first; graphs are not a default, they are a targeted fix.
- Your **shapes change every request** and you cannot or will not bucket them. A graph per shape has real memory and complexity cost; if the shape space is large and unbucketable, graphs may not be worth it.
- The region **needs host-side control flow** — data-dependent early exit, dynamic loop counts, mid-region CPU reads — that you cannot hoist out. Fighting the frozen-DAG contract is usually a sign the region should not be graphed.
- You have **not measured yet.** The cardinal sin. If you do not know your `N`, your `t_launch`, and your `Σt_gpu`, you do not know whether the speedup is 2.8× or 1.02×. Measure, then decide.

The honest framing: graphs are one of four fixes for four different wastes. Launch overhead is one waste; graphs kill it. Redundant work is another — that is fusion and `torch.compile`. Memory-bound kernels are another — that is the memory-format and bandwidth work in Track E. Idle-from-the-CPU-side dataloader starvation is another — that is Track F. The [capstone playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) is the decision tree that routes a symptom to the right one of these. A graph is the right tool for exactly one symptom, and applying it to the others wastes your time and their flexibility.

## Key takeaways

- A CUDA graph is a **recorded DAG** of GPU operations — kernels, copies, memsets — with their launch parameters, dependencies, and memory addresses **frozen** into one replayable object.
- The lifecycle is **warm up → capture → instantiate → replay**. Capture records without committing; instantiate compiles the template once; replay fires the whole DAG in a single driver call.
- The mechanism is a **host-cost collapse**: eager pays `N · t_launch` per step, graphed pays about one launch per step. The speedup factor is `N`, capped by how launch-bound you were — `speedup ≈ t_step^eager / Σt_gpu`.
- Graphs are a **dispatch optimization, not a compiler**. They make launching kernels nearly free; they do not fuse, rewrite, or speed up any individual kernel. Compose them with `torch.compile` to get both.
- The price is **rigidity**: static shapes, static memory addresses, no data-dependent control flow, no in-region host synchronization. That rigidity is exactly what makes replay cheap.
- The scariest failures are **silent** — a moved pointer or a changed shape produces wrong output with no error, because the GPU faithfully replays the frozen (now-wrong) plan. Always assert numerical equality against the eager path.
- The win is biggest for **many tiny kernels** — decode at low batch, elementwise chains — and near zero for a single big GEMM. Profile for the launch-bound signature before you graph.
- **Measure honestly**: warm up, synchronize or use CUDA events, feed fresh data into the static input on every replay, and never fold capture cost into the per-step average.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes and the profile-hypothesize-fix-measure loop this whole series runs on.
- [The kernel launch overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — the previous post: measuring `t_launch`, spotting the launch-bound signature, and why tiny kernels starve the GPU.
- [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — the concrete API behind everything here: `torch.cuda.graph`, `make_graphed_callables`, the graph pool, static I/O tensors.
- [CUDA graphs gotchas and debugging](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) — what to do when a gate fails: dynamic shapes, allocator interactions, in-region sync, and debugging silent garbage output.
- [Kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — how graphs slot into a production serving stack alongside fusion and compilation.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree that routes each symptom to the right fix.
- NVIDIA CUDA C++ Programming Guide, "CUDA Graphs" section, and the CUDA Runtime API docs for `cudaStreamBeginCapture`, `cudaGraphInstantiate`, and `cudaGraphLaunch` — the primary source for the raw stream-capture model.
- PyTorch docs: "CUDA semantics — CUDA Graphs" and the `torch.cuda.graph` / `make_graphed_callables` API reference for the framework wrapper around everything above.
