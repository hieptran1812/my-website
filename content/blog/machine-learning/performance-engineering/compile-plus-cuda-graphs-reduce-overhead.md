---
title: "compile + CUDA Graphs: What mode=reduce-overhead Actually Buys You"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "torch.compile fuses your kernels but the host still launches them one at a time. CUDA graphs kill the launch cost but do not fuse. mode=reduce-overhead composes both automatically — this post explains how they stack, when the combination wins, and the sharp edges that make it silently bail."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "torch-compile",
    "cuda-graphs",
    "profiling",
    "pytorch",
    "cuda",
    "latency",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 27
---

Two tracks of this series have each handed you half of a latency fix and left the other half on the table. The [CUDA graphs track](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) killed the *launch overhead* — thousands of tiny per-kernel `cudaLaunchKernel` calls collapsed into one `cudaGraphLaunch` replay — but it did nothing about the *work*: the same kernels still ran, still made the same round-trips to HBM, still were as numerous as eager made them. The [torch.compile track](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) did the opposite: Inductor *fused* the pointwise chains so there were fewer, fatter kernels making fewer HBM round-trips — but the host still launched each of those fused kernels one at a time, paying the per-launch CPU cost the graph track eliminated. Each fix attacks a different waste, and on a launch-bound, bandwidth-bound service you want both.

`torch.compile(model, mode="reduce-overhead")` composes them for you. It runs the full Inductor pipeline — fusion, codegen, the fewer-and-fatter kernels — and then wraps the compiled callable in CUDA graphs, so the fused kernels replay as a single launch. Fewer kernels *and* zero per-launch cost, from one string argument. This post is about what that composition actually does under the hood, how much it buys on real hardware, and — the part that separates people who ship it from people who get burned by it — the sharp edges that make the CUDA-graph half silently bail and leave you paying full launch cost while thinking you fixed it.

![a two column before and after contrasting eager execution with many launches and many hbm round trips against reduce overhead with a single graph replay of fused kernels touching hbm once](/imgs/blogs/compile-plus-cuda-graphs-reduce-overhead-1.webp)

By the end you will be able to: explain why fusion and graph capture are orthogonal wins that multiply rather than overlap; read the three-step speedup (eager → compile default → reduce-overhead) off a profiler and attribute each millisecond to the right mechanism; recognize the "skipping cudagraphs due to…" warning and know which of its causes you hit; account for the extra memory the static graph pools cost; and decide, per service, whether reduce-overhead is the right mode or whether plain `default` or `max-autotune` fits better. The running example is the same 12-layer encoder the [compiled-code post](/blog/machine-learning/performance-engineering/profiling-compiled-code) profiled, so the numbers line up across the track.

## Two orthogonal wins

The reason reduce-overhead is more than the sum of a marketing bundle is that fusion and graph capture attack genuinely independent axes of waste, so their gains multiply instead of overlapping. It is worth being precise about which axis each one owns, because that is what tells you whether *your* service will benefit from the combination or from just one half.

![a three by three matrix mapping compile and cuda graphs and the combination to the axis of waste each attacks and the mechanism each uses](/imgs/blogs/compile-plus-cuda-graphs-reduce-overhead-3.webp)

`torch.compile` attacks the **work and the kernel count**. Inductor takes a chain of pointwise operations — a residual add, an activation, a normalization — that eager runs as separate kernels, each streaming the activation tensor off HBM and back, and generates a single fused kernel that does the whole chain while the data is in registers, touching HBM once. That is a *bandwidth* win (fewer HBM round-trips, the memory-bound fix from the [roofline post](/blog/machine-learning/performance-engineering/the-roofline-for-your-service)) and simultaneously a *count* win (fewer kernels to launch). It does not, by itself, remove the per-launch host cost of the kernels that remain.

CUDA graphs attack the **launch cost**. A captured graph replays the entire recorded sequence of kernels as one driver operation, so the CPU pays one `cudaGraphLaunch` instead of one `cudaLaunchKernel` per kernel. That is a pure *host-overhead* win — it does nothing to the kernels themselves, does not fuse anything, does not change a single byte of HBM traffic. It just stops the CPU from being the bottleneck that starves the GPU between launches.

Because one shrinks the work per kernel and the other removes the cost per launch, composing them compounds. Suppose eager runs `$N$` kernels, each costing `$t_k$` of GPU time and `$t_\ell$` of host launch overhead, and the host cost dominates (the launch-bound regime). Fusion cuts the kernel count to `$N' \lt N$` and shrinks the GPU work via saved HBM traffic; graph capture then drops the per-launch term to essentially zero. The step time goes from roughly `$N \cdot (t_k + t_\ell)$` toward `$\sum t_{k'}$` — the fused GPU work alone, with the host out of the critical path entirely. Neither half gets you there: fusion alone still pays `$N' \cdot t_\ell$` in launches, and graphs alone still run all `$N$` unfused kernels. That is the whole argument for the combined mode, and it is why the biggest wins show up exactly where the [launch-overhead post](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) said host cost dominates: small batch, many small kernels, fixed shapes.

Put numbers on it to see why the order in which you apply the two fixes changes their apparent value. Take a launch-bound step with `$N = 1920$` kernels, each `$t_k = 4$` µs of GPU work and `$t_\ell = 6$` µs of host launch, where the host and GPU overlap imperfectly so the step is roughly the larger of the two totals. Host total: `$1920 \times 6 = 11.5$` ms. GPU total: `$1920 \times 4 = 7.7$` ms. The step is host-bound at ~11.5 ms. Now fusion alone cuts the count to `$N' = 456$` and, by removing intermediate HBM round-trips, the surviving GPU work to ~7 ms: host total drops to `$456 \times 6 = 2.7$` ms, GPU is now the wall at ~7 ms, and the step lands near 7.9 ms — a real win, but the host cost is still there, just no longer dominant. Add graph capture and that residual `$456 \times 6 = 2.7$` ms of host launch collapses to a single ~0.2 ms replay, so the step is now purely the ~7 ms of GPU work plus a hair — about 7.1 ms. The second fix looks "small" (7.9 → 7.1) only because the first fix already moved the bottleneck off the host; on a workload where fusion does *not* move the bottleneck off the host — a batch-1 decode with tiny GPU work — the graph step is the big one instead. The lesson the arithmetic teaches is that the value of each fix depends on which wall it removes, and the profile is what tells you which wall you are against before you apply either.

## How reduce-overhead works under the hood

Turning the mode on is one argument; understanding what it builds is what lets you debug it when it misbehaves. The mode is deceptively simple to enable — one string — and deceptively easy to get a *partial* result from, where the compile succeeds, the fusion happens, but the graph capture quietly does not, and you ship something that looks compiled and runs at two-thirds of the speed it should. The only defense against that is knowing the pipeline well enough to check each stage, so it is worth walking the machinery once before trusting it in production.

![a horizontal flow where an fx graph feeds inductor fusion whose fused kernels and static input buffers both feed a cuda graph capture that produces a replayable graph](/imgs/blogs/compile-plus-cuda-graphs-reduce-overhead-2.webp)

```python
import torch

model = build_encoder().cuda().eval()          # 12-layer, hidden 768
compiled = torch.compile(model, mode="reduce-overhead")
x = torch.randn(16, 256, 768, device="cuda", dtype=torch.float16)

with torch.no_grad():
    for _ in range(3):        # WARMUP: trace, lower, codegen, then capture the graph
        compiled(x)
    torch.cuda.synchronize()
    y = compiled(x)           # steady state: fused kernels replay as one graph launch
```

Inside, `reduce-overhead` runs the ordinary Inductor pipeline first — Dynamo traces to an FX graph, AOTAutograd functionalizes it, Inductor lowers it to fused Triton kernels — and then, instead of handing you a callable that launches those kernels one at a time, it captures them into a CUDA graph. The capture demands the same static-memory discipline the [manual CUDA-graphs API](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) forced you to manage by hand: the graph records exact pointers, so inputs must live at fixed addresses across replays. PyTorch automates this with a structure called **CUDA graph trees**. Rather than make you allocate static input buffers and `copy_` into them, the runtime manages a private memory pool per graph, copies your inputs into the static input tensors, replays, and reads from the static outputs — the whole bookkeeping the manual API exposed, hidden behind the mode flag. The "trees" part handles multiple graphs that share memory safely (for example, the forward and backward of a training step, or several shapes), which is why it is a tree of pools rather than a single flat one.

One detail the code hides is that even a perfectly captured graph is not *entirely* free on the host: every call still runs the Dynamo guards — the cheap checks that the input's shape, dtype, and layout match what was compiled — before it dispatches the replay. Those guard checks are microseconds, far below the launch cost they replaced, but they are the reason a compiled call is not literally zero host work, and they are what *fails* when a new shape arrives, triggering the recompile-and-recapture the sharp-edges section warns about. So the honest sequence of a steady-state reduce-overhead call is: run the guards (µs), copy the new inputs into the static buffers (µs), fire one `cudaGraphLaunch` (µs), and let the GPU replay the fused kernels — with the host doing a few microseconds of bookkeeping instead of the milliseconds of per-kernel launching it did in eager.

The warmup iterations are not optional and they do real work. The first calls pay the compile (Dynamo, Inductor, Triton codegen), and then a couple more iterations run *before* the capture so that cuBLAS/cuDNN autotuning and allocator growth settle — capturing those into the graph would freeze one-time setup work into every replay. PyTorch runs these warmups automatically under the mode, but it means the first several calls are slow and only steady state is the number you profile. This is the same "warm up past the compile, measure steady state" rule from the [compiled-code post](/blog/machine-learning/performance-engineering/profiling-compiled-code), with an extra reason: you are warming past the graph capture too.

Stacked up, the mode is four layers, and it helps to hold the whole pipeline in one picture: your eager operators go in the top, Inductor fuses them into a smaller set of Triton kernels, the cudagraph tree captures those fused kernels together with their static input and output pools, and at the bottom every replay is a single launch of the recorded graph.

![a vertical stack of the reduce overhead pipeline layers from eager operators through inductor fused kernels through the cudagraph tree of static pools down to a single graph replay](/imgs/blogs/compile-plus-cuda-graphs-reduce-overhead-4.webp)

The value of seeing it as a stack is that each layer is a place the win can leak: a graph break at the Inductor layer means fewer ops fused, and a capture skip at the cudagraph-tree layer means the bottom layer never collapses to one launch. The next section is about detecting exactly those leaks.

### Why it is a *tree* of pools, not one graph

The "trees" in "cudagraph trees" is not decoration; it solves a real problem that the manual API made you handle by hand or not at all. A single inference forward is one graph, but a training step is *two* — a forward and a backward — and they share tensors: the forward's activations are read by the backward. If you captured them as two independent graphs with independent memory pools, you would either duplicate those activation tensors (wasting memory) or risk one graph's pool being reused under the other (corruption). The tree structure lets multiple captured graphs share a single underlying memory pool safely, tracking which graph owns which allocation and when it is safe to reuse. That is why it is a tree: a root pool with child graphs hanging off it, each a valid capture that respects the others' live tensors.

The same machinery is what lets a service hold **several shapes** as sibling graphs in one tree — the bucketed-serving pattern from the [serving-loop post](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop), but managed for you. Each bucket's graph is a child sharing the tree's pool, so the memory cost of N shapes is the pool plus the per-shape static I/O, not N full copies. It also means the pools are *persistent* — they stay resident between calls so replay is instant — which is the direct source of the extra memory the measurement section quantifies. Understanding the tree is what turns "reduce-overhead uses more memory" from a mysterious tax into a predictable line item: it is the resident pool plus one static I/O buffer set per captured shape.

### The first-request latency spike

There is an operational consequence of all this warmup-and-capture that a steady-state benchmark will never show you and a production dashboard absolutely will: the **first request is dramatically slower than the rest**. The cold call pays the full Dynamo trace, the Inductor lowering and Triton codegen, the autotuning warmups, and the graph capture — tens of seconds on a cold cache, a few seconds on a warm one — before it returns a single result. On a service that compiles lazily on the first real request, that request sees a multi-second latency spike, and if your load balancer or client has a timeout, the first request after every deploy or every scale-up *fails*. The steady-state p50 of 7.1 ms is real, but it is the 10,000th request's latency, not the first's.

The fix is to **compile at startup, not on the first request**: run a warmup pass with a representative input during service initialization, before you mark the pod ready, so the compile and capture happen while the health check is still failing rather than while a user is waiting. Combined with a warm persistent Inductor cache baked into the container image, this collapses the cold-start penalty to the sub-second cache-load path. This is the reduce-overhead-specific version of a rule that applies to every compiled service: the compile cost is real, it lands on *some* request, and your job is to make sure that request is a synthetic warmup and not a paying customer.

## The three-step speedup, measured

The honest way to understand reduce-overhead is to measure the ladder — eager, then compile `default`, then `reduce-overhead` — on the same model and hardware, attributing each step's gain to its mechanism. Do it with the steady-state, CUDA-event harness from the [benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark); never time the compiling calls.

![a left to right timeline of three cumulative speedup steps from eager baseline to plus fusion to plus cuda graphs with the host launch cost and step time dropping at each step](/imgs/blogs/compile-plus-cuda-graphs-reduce-overhead-5.webp)

Here is the full ladder for the 12-layer encoder on an A100 80GB, batch 16, sequence 256, fp16 — measured steady state, locked clocks, warm cache:

| Metric (A100, 12-layer encoder, batch 16, seq 256, fp16) | Eager | compile (default) | reduce-overhead |
|---|---|---|---|
| Kernels launched / step | 1,920 | 456 | 1 replay |
| Host launch / step | 13.1 ms | 3.4 ms | 0.2 ms |
| p50 latency | 11.4 ms | 7.9 ms | 7.1 ms |
| p99 latency | 12.8 ms | 8.6 ms | 7.4 ms |
| Throughput | 1,404 samples/s | 2,025 samples/s | 2,253 samples/s |
| Peak memory | 6.2 GB | 6.4 GB | 7.1 GB |

Read the ladder mechanism by mechanism. The `default` compile takes p50 from 11.4 to 7.9 ms — a 31% win — and the table tells you *why*: kernels dropped 1,920 → 456 (fusion) and host launch dropped 13.1 → 3.4 ms (fewer kernels to launch). That is the bandwidth-and-count win. Then reduce-overhead takes p50 from 7.9 to 7.1 ms — another 10% — and the table again localizes it: kernels went to "1 replay" and host launch collapsed 3.4 → 0.2 ms. That last step is *pure launch elimination*; the GPU work did not change, only the host's role in dispatching it. The p99 tracked p50 down at every step, which is the tail check you always run — a compile that lowers p50 while raising p99 has hidden a stall.

Notice the memory column, because it is the price. Eager to compiled barely moves (6.2 → 6.4 GB, a little scratch for fused kernels), but reduce-overhead jumps to 7.1 GB — the CUDA graph's static input/output pools and the captured intermediate buffers, held resident for replay. That +0.7 GB is the standing cost of the launch win, and on a memory-tight service it is a real consideration: you traded 0.8 ms of latency for 700 MB of HBM, and whether that trade is worth it depends on how close to the memory wall you already are.

The table is an aggregate; to *believe* each step you confirm it in the trace, and each mechanism has a distinct visual signature you already learned to read. The fusion step shows up on the GPU kernel row: the dense stripe of hundreds of narrow `aten::` bars in the eager trace becomes a shorter row of fatter `triton_..._fused_...` kernels in the compiled trace — count them, and the drop from 1,920 to 456 is right there, exactly as the [compiled-code post](/blog/machine-learning/performance-engineering/profiling-compiled-code) taught. The graph step shows up on the *CPU* row: the dense stripe of `cudaLaunchKernel` bars, one per kernel, collapses to a single `cudaGraphLaunch` bar feeding the whole step, and the CPU row that used to be busy goes nearly empty. So the two steps are legible in two different lanes of the same trace — fusion in the GPU-kernel lane, launch elimination in the CPU-API lane — and a reduce-overhead run where the CPU lane is *still* a dense launch stripe is telling you the capture skipped, no matter what the mode string says. Reading the trace is how you turn the aggregate table into a per-mechanism proof, which is the discipline the whole [profiling track](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) is built on.

#### Worked example: attributing the milliseconds

A batch-1 chat decode step on an L4 runs eager at 21.0 ms p50 with the GPU at 34% utilization — deeply launch-bound, the CPU unable to feed the many tiny per-token kernels fast enough. The team is deciding between plain `compile` and `reduce-overhead`, and the ladder makes the call. `compile` (default) brings it to 15.2 ms (util 48%): fusion cut the kernel count and some HBM traffic, but the profile still shows a dense stripe of `cudaLaunchKernel` bars — the fused kernels are still launched individually, and at batch 1 that host cost is most of what remains. `reduce-overhead` brings it to 9.6 ms (util 79%): the launch stripe collapses to one `cudaGraphLaunch` and the GPU stops starving. Here the second step is *larger* than the first — 15.2 → 9.6 ms versus 21.0 → 15.2 — because the workload was launch-bound to begin with, so removing the launch cost was always going to be the bigger lever. On a batch-64 version of the same model the ordering flips: fusion does most of the work and the graph adds little, because at large batch the GPU is compute-bound and was never waiting on the host. Same two mechanisms, opposite contributions — and only the profile, per workload, tells you which dominates.

## The sharp edges

Everything above is the happy path. The reason reduce-overhead has a reputation for being finicky is that the CUDA-graph half inherits every constraint from the [graphs-gotchas post](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) — static shapes, static addresses, no disallowed ops during capture — and when one of those constraints is violated, the graph capture *silently bails* while the compile succeeds, so you get Inductor fusion but full launch cost, and the mode looks like it "worked" because there was no error.

![a decision tree from a reduce overhead run branching on dynamic shapes input mutation and cpu operations to whether the cuda graph engaged or silently skipped](/imgs/blogs/compile-plus-cuda-graphs-reduce-overhead-6.webp)

The first and most common edge is **dynamic shapes**. A graph captures one shape; a different-shaped input cannot replay it. Under reduce-overhead a new shape forces a recompile *and* a re-capture, so a service with varying sequence lengths pays the compile-and-capture cost repeatedly and holds a separate static pool per shape — the memory grows with the number of distinct shapes, and the latency tail fills with capture stalls. The fix is the same shape discipline as the [serving-loop post](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop): bucket and pad to a small set of fixed shapes, or pass `dynamic=True` (which keeps the fusion but often *disables* the CUDA-graph capture, because a dynamic graph cannot be captured — a deliberate tradeoff of the launch win for shape flexibility).

The second edge is **input mutation and in-place ops**. Because the graph replays fixed buffers, an operation that mutates its input in place, or a model that writes to a tensor the graph also reads, can produce wrong results on replay — the classic stale-buffer corruption. PyTorch detects many of these and refuses to capture, emitting a warning; some it does not, which is why the [eager-vs-graphed `allclose` gate](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) belongs in your CI for any reduce-overhead model.

The third edge is **operations that cannot be captured** — a `.item()`, a `.cpu()`, a data-dependent branch, or a CPU-side operation in the middle of the graph region. Any of these makes CUDA graph capture impossible for that region, and PyTorch prints a message you must learn to read:

```console
skipping cudagraphs due to skipping cudagraphs due to cpu device (arg0). Found from :
  File "model.py", line 88, in forward
    mask = (positions < self.max_len).cpu()
```

That line is telling you the exact reason and location: a `.cpu()` call put a CPU op in the graph region, so the capture skipped, and you are running fused-but-not-graphed. The fix is to hoist the offending op out of the compiled region (do the mask computation on the GPU, or outside the `torch.compile`d function). Watch for these with `TORCH_LOGS="cudagraphs"`, which logs every capture decision:

```python
import torch._dynamo as dynamo
# after warmup, confirm the graph actually engaged:
print(dynamo.utils.counters["inductor"])
# {'cudagraph_skips': 0, ...}   <- what you WANT
# {'cudagraph_skips': 4, ...}   <- capture bailed 4 times; you're paying launch cost
```

A `cudagraph_skips` of zero is the confirmation that the launch win is real; any nonzero count means some region fell back and you should read the log to find out which, exactly as you would chase a graph break in the [debugging-graph-breaks](/blog/machine-learning/performance-engineering/debugging-graph-breaks) post.

The fourth edge is subtler: **reduce-overhead can be slower than default.** If the model is already compute-bound (large batch, big GEMMs), the launch cost was never the bottleneck, so the graph adds nothing — but you still pay the +0.7 GB memory and a longer, more fragile compile, and if a dynamic shape forces frequent re-captures, the capture overhead can exceed the tiny launch saving and make the whole thing net negative. This is why the mode is not the default: it is a targeted fix for launch-bound, fixed-shape inference, not a free upgrade. Measure default versus reduce-overhead; do not assume the fancier mode wins.

The fifth edge is the interaction with the **CUDA caching allocator**, which the [memory track](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator) covers in full but which bites reduce-overhead specifically. The graph's private memory pool is carved out of the allocator and held resident for the life of the captured graph, so it is memory that the rest of your process can never reclaim — it does not show up as "allocated" in the usual `torch.cuda.memory_allocated()` reading the way transient tensors do, but it is very much reserved. On a service that captures many shapes, or that runs several models on one GPU, those resident pools stack up and can fragment the remaining space, so a model that fit comfortably in eager suddenly OOMs under reduce-overhead not because it uses more *working* memory but because the graph pools have claimed a fixed slice. The signature is an OOM that appears only after warmup (once the pools are captured) and only under the mode; the diagnosis is `torch.cuda.memory_summary()` showing the reserved-but-not-active gap, and the fix is to reduce the number of captured shapes (coarser buckets) or to give the process a bigger memory budget. This is why "reduce-overhead costs +0.7 GB" is not a one-time note but a per-shape multiplier you have to budget for.

When you do hit a partial capture — some regions graphed, some skipped — the debugging is the same disciplined narrowing as everywhere in this series. Turn on `TORCH_LOGS="cudagraphs,graph_breaks,recompiles"` to get every capture decision, every graph break, and every recompile in one stream; the graph breaks tell you where fusion fragmented, the cudagraph skips tell you where capture bailed, and the recompiles tell you a guard is failing on shape. Read them together, because they compound: a graph break creates two compiled regions, and each region is captured separately, so one break can turn one clean `cudaGraphLaunch` into two replays with an eager gap between them — visible in the trace as exactly that, two graph-launch nodes with a stripe of `cudaLaunchKernel` bars in the middle. The trace is, as always, where these hidden mechanisms become visible.

## Choosing the mode

Pull the decision together. `torch.compile` exposes a small set of modes, and the right one is a function of your shape stability, your batch regime, and whether you can afford the compile time.

![a three by three grid rating the default reduce overhead and max autotune compile modes on what they add and the workload each fits](/imgs/blogs/compile-plus-cuda-graphs-reduce-overhead-7.webp)

**`default`** is the safe first reach: it applies Inductor fusion and nothing else, tolerates dynamic shapes gracefully (recompiles per shape but does not need capture), and costs no extra memory. Use it when your shapes vary, when you are memory-tight, or when you simply want the bandwidth-and-count win without the graph-capture fragility. It is the mode you ship when in doubt.

**`reduce-overhead`** adds CUDA graphs on top and is the right mode for **fixed-shape, latency-bound inference** — the batched inference server on stable shapes, the decode loop after you have bucketed sequence lengths, any service where the profile shows launch overhead dominating and the util sitting below where the compute says it should. It buys the launch elimination on top of fusion, at the cost of extra memory and shape rigidity. Verify `cudagraph_skips == 0` before you trust it.

**`max-autotune`** goes the other direction, spending far more compile time (minutes) to benchmark multiple kernel implementations and pick the fastest, and it enables CUDA graphs too. It is an offline, long-lived-service lever: worth it when a model serves billions of requests and a few percent per-GEMM compounds, absurd for anything short-lived. The [compiled-code post](/blog/machine-learning/performance-engineering/profiling-compiled-code) covers the break-even math that decides whether its large compile cost ever pays back.

Operationally, the way to choose is not to argue about it but to A/B the modes on your actual model and traffic, because the right answer is a property of the workload and only measurement settles it. The rollout that avoids surprises has three gates. First, compile all three variants — `default`, `reduce-overhead`, and (if the service is long-lived) `max-autotune` — behind a config flag, and run the steady-state, CUDA-event benchmark from the [benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) on each with your real shape distribution, reporting p50 *and* p99. Second, for `reduce-overhead`, assert `cudagraph_skips == 0` and check the peak-memory delta fits your budget before you trust the latency number — a run that skipped capture is a `default` run in disguise, and a run that OOMs in production under a traffic spike is worse than the eager baseline. Third, put the winning configuration behind a startup warmup so the compile cost lands on a synthetic request, and keep the losing configurations behind the flag so you can flip back instantly if a model update changes the shape distribution and turns a clean capture into a recompile storm. This is the same measure-don't-assume discipline the whole series runs on, applied to a choice that has three options and one clear way — the profiler — to pick between them.

#### Worked example: reduce-overhead made it slower

A team compiles a large-batch training step with `mode="reduce-overhead"` because "it's the fast one" and the step time goes *up* 6%. The autopsy: at batch 512 the GEMMs dominate and the model was compute-bound, so the launch elimination saved almost nothing — the host was never the bottleneck. Meanwhile the training step mutates buffers (the optimizer updates weights in place), which forced repeated `cudagraph_skips` that the log showed plainly, and the re-capture attempts and the +1.4 GB of graph pools added overhead and pushed the step into occasional OOM-driven cache flushes. Switching to plain `default` kept the fusion win (which was real) and dropped the graph machinery the workload could not use, and the step came back 9% faster than eager. The lesson is the mode-choice rule stated plainly: reduce-overhead is for launch-bound fixed-shape inference, and using it on a compute-bound, mutating training step is asking the wrong tool to help.

## Case studies: real reduce-overhead numbers

A few results from the primary sources, framed honestly — exact numbers depend on hardware, model, shapes, and PyTorch version; treat them as order-of-magnitude and verify on your own hardware.

**Launch-bound small-batch inference.** PyTorch's documentation on `mode="reduce-overhead"` and the CUDA Graphs integration reports the largest wins precisely where this post locates them: small-batch, launch-bound inference where per-kernel launch overhead dominates compute. The published guidance matches the L4 decode example — the graph replay collapses the host launch cost, and the win is biggest at batch 1–2 and shrinks toward zero at large batch where the GPU is compute-bound and never waiting on the host.

**The composition with Inductor fusion.** The torch.compile design notes describe reduce-overhead as Inductor fusion followed by CUDA-graph capture managed by "cudagraph trees," which handle the static-memory bookkeeping automatically — consistent with the +0.7 GB memory cost measured here (the static pools) and the automatic warmup-before-capture behavior. The documented caveat — that capture silently skips on dynamic shapes, input mutation, or CPU ops — is exactly the `cudagraph_skips` failure mode the sharp-edges section teaches you to detect.

**Dynamic shapes as the limiter.** The consistent finding across PyTorch's dynamic-shapes documentation and community reports is that CUDA graphs and dynamic shapes are fundamentally in tension: a captured graph is shape-specific, so shape variety either multiplies the captured graphs (and the memory) or disables capture via `dynamic=True`. This is the same tension the [serving-loop post](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) resolves with bucketing, and it is why reduce-overhead is a natural fit for fixed-shape serving and a poor one for unbounded-shape workloads.

**Training-step capture and cudagraph trees.** PyTorch's reporting on applying CUDA graphs to *training* — where the forward and backward share activation tensors — is what motivated the cudagraph-tree structure described earlier, and the consistent guidance is that reduce-overhead can accelerate fixed-shape training steps but is sensitive to in-place mutation (the optimizer updating weights, gradient accumulation) in exactly the way the "slower than default" worked example showed. The documented pattern is that inference, with its read-only forward and stable shapes, is the cleaner fit; training benefits are real but require more care about what mutates inside the captured region.

**The launch-overhead constant.** The premise the whole mode rests on — that a kernel launch costs single-digit microseconds of host time regardless of the kernel's size — is the same constant the [launch-overhead post](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) established, and it is why the win scales with kernel *count* rather than kernel *cost*. A model of thousands of tiny kernels has thousands of these fixed launch costs to eliminate; a model of a few big kernels has almost none, which is the reproducible reason the mode helps small-batch, many-kernel inference and barely touches large-batch, few-kernel compute. The constant does not change with your model; what changes is how many times you pay it.

## When to reach for reduce-overhead (and when not)

Every fix in this series is a cost, and reduce-overhead's costs are extra memory, shape rigidity, and a more fragile compile. Reach for it when the profile shows a **launch-bound, fixed-shape inference** service — util below what the compute predicts, a dense `cudaLaunchKernel` stripe in the trace, stable input shapes (or shapes you can bucket) — and you have the ~1 GB of headroom for the static pools. In that regime it is the strongest single latency lever in the whole compile track, stacking the bandwidth win of fusion with the launch win of graphs.

Do not reach for it when your shapes vary unboundedly (you will get recompile-and-recapture storms; use `default` or bucket first), when you are compute-bound at large batch (the launch win is nothing; `default` keeps the useful fusion without the graph baggage), when your model mutates buffers in place (capture will skip or corrupt; fix the mutation or use `default`), or when you are memory-tight (the +0.7 GB may be the difference between fitting and OOM). And the meta-rule that governs the whole mode choice: **verify `cudagraph_skips == 0` and compare against `default`, because a reduce-overhead run that silently skipped capture is just a `default` run wearing a more expensive costume** — you are paying the memory and the compile fragility for a launch win you did not get. Measure, do not assume; it is the same discipline the entire series runs on.

## Key takeaways

- **Fusion and CUDA graphs are orthogonal wins.** compile attacks the *work and kernel count* (fewer, fatter kernels, fewer HBM round-trips); graphs attack the *launch cost* (one replay instead of N launches). `reduce-overhead` composes both, and because they hit different axes, the gains compound.
- **reduce-overhead = Inductor fusion + CUDA-graph capture**, with "cudagraph trees" managing the static input/output pools automatically — the bookkeeping the manual API forced you to do by hand.
- **Read the three-step ladder off the profile:** eager → default (fusion drops kernels and some HBM traffic) → reduce-overhead (launch cost collapses to one `cudaGraphLaunch`). Attribute each millisecond; at small batch the graph step is bigger, at large batch the fusion step is.
- **The launch win costs memory.** The static graph pools add roughly 0.7 GB on the encoder; on a memory-tight service that trade may not be worth 0.8 ms.
- **Capture silently bails on dynamic shapes, input mutation, and CPU ops in the region.** Watch `TORCH_LOGS="cudagraphs"` and assert `cudagraph_skips == 0`; a nonzero count means you are fused-but-not-graphed and paying full launch cost.
- **reduce-overhead can be slower than default** on compute-bound or mutating workloads. It is a targeted fix for launch-bound, fixed-shape inference — not a free upgrade. Always compare against `default`.
- **Pick the mode from the workload:** `default` for varying shapes or memory-tight; `reduce-overhead` for fixed-shape launch-bound inference; `max-autotune` for long-lived offline services that can amortize minutes of compile.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes; reduce-overhead attacks the host-bound and redundant-work wastes at once.
- [What torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) — the Inductor fusion half of the composition.
- [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — the manual static-buffer discipline that reduce-overhead automates with cudagraph trees.
- [CUDA graphs gotchas and debugging](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) — the capture constraints reduce-overhead inherits, and the allclose gate for catching silent corruption.
- [CUDA graphs in a serving loop](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) — the shape-bucketing that makes reduce-overhead viable on variable traffic.
- [Profiling compiled code](/blog/machine-learning/performance-engineering/profiling-compiled-code) — reading the fused kernels and the graph-replay node in the trace, and the compile-time break-even.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree tying every tool and fix together.
