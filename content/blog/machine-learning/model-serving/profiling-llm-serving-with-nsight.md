---
title: "Profiling LLM Serving with Nsight: Finding the Real Bottleneck at the Kernel Level"
date: "2026-07-07"
publishDate: "2026-07-07"
description: "A hands-on, data-driven guide to profiling an LLM server from coarse Prometheus metrics down to a single CUDA kernel — reading the Nsight Systems timeline and Nsight Compute report to find whether you are launch-, memory-, or compute-bound, and fixing the one that matters."
tags:
  [
    "model-serving",
    "inference",
    "profiling",
    "nsight-systems",
    "nsight-compute",
    "torch-profiler",
    "gpu-optimization",
    "cuda",
    "roofline",
    "ml-infrastructure",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/profiling-llm-serving-with-nsight-1.webp"
---

The worst way to optimize an LLM server is to guess. I have watched a team spend two sprints on a fancier batching scheduler because "the GPU looked underutilized," only to discover — after finally attaching a profiler — that the scheduler was already near-optimal and the real cost was 300 tiny CUDA kernels per decode step that left the GPU waiting on the CPU a third of the time. The fix was one flag (`enforce_eager=False`, to turn CUDA graphs back on) and it took an afternoon. Two sprints of work were aimed at the wrong number because nobody had looked at the actual kernel timeline.

This post is about looking. It is the hands-on craft of profiling an LLM inference server at the level where the truth lives: the CUDA kernels, the gaps between them, and the hardware counters inside them. The goal is never "make it faster" in the abstract — it is to answer one specific question, **what is this workload actually bound by right now**, and then to spend your engineering time on that and nothing else. There are exactly three answers that matter for a serving workload — you are *launch-bound* (the GPU is idle waiting for the CPU to issue work), *memory-bound* (the GPU is saturating HBM bandwidth and starving the compute units), or *compute-bound* (the tensor cores are actually busy) — and a good profile tells you which one in minutes.

![A four-rung layered stack showing the profiling ladder from fleet-wide server metrics down through torch.profiler and Nsight Systems to a single kernel in Nsight Compute, with the overhead rising at each rung](/imgs/blogs/profiling-llm-serving-with-nsight-1.webp)

The figure above is the spine of this post: a **profiling ladder**. You start at the top rung, which is nearly free — the Prometheus metrics your server already emits — and you descend one rung at a time, paying more overhead and narrowing the question at each step, but *only* when the rung above has pointed somewhere specific. Rung two is the PyTorch profiler, which localizes the hot region inside one process. Rung three is Nsight Systems (`nsys`), which draws the kernel-by-kernel timeline and shows you the gaps, the CPU-GPU overlap, and the NCCL communication. Rung four is Nsight Compute (`ncu`), which puts a single kernel under a microscope and reports its SM occupancy, DRAM throughput, and tensor-core utilization. By the end you will be able to run each rung against a live vLLM server, read the artifacts it produces, map any kernel to its place on the roofline, and turn a symptom in the trace into a specific fix.

Every one of these decisions is a trade on the serving SLO triangle — **latency ↔ throughput ↔ cost**. Profiling does not change the triangle; it tells you which corner you are actually paying for, so you stop trading away the wrong one. If you have not internalized why LLM decode is bandwidth-limited in the first place, the companion post on [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) covers the KV-cache memory wall that makes the whole autoregressive phase memory-bound; this post assumes that background and goes straight to measuring it.

## Profile the steady state, not the cold start

Before a single command, the most important discipline in profiling a *serving* workload — as opposed to a training run or a microbenchmark — is knowing *what window to capture*. Get this wrong and every number you collect is a lie. A serving process has three distinct regimes, and only one of them is the thing you are being paged about.

The first regime is **cold start**: the server loads weights, allocates the KV-cache blocks, and runs a few dummy forward passes. During cold start the GPU is doing memory allocation, `cudaMalloc`, cuBLAS/cuDNN algorithm selection (the first call to a GEMM shape triggers an autotuning search that can take tens of milliseconds), and — critically — CUDA-graph capture if the framework uses it. None of these happen on the steady-state path. If you profile the first ten seconds of a vLLM server, most of your timeline will be one-time setup that never recurs, and you will "optimize" things that run exactly once per process lifetime.

The second regime is **warmup**: the first several hundred iterations after cold start. Here the GPU clocks are still ramping (a modern data-center GPU boosts its clock under sustained load and throttles when idle), the memory allocator's caching layer is still filling, the OS page cache is cold, and — in PyTorch — the autograd/dispatcher caches and any `torch.compile` guards are still being populated. `torch.compile` in particular will *recompile* on the first few shapes it sees, and those recompilations show up as enormous CPU stalls that have nothing to do with your steady-state cost. Numbers collected during warmup are systematically pessimistic and noisy.

The third regime is **steady state**: the server has been serving representative traffic long enough that clocks are boosted, caches are warm, graphs are captured, and every iteration looks like every other iteration. **This is the only regime worth profiling**, because it is the only one that recurs under production load. The entire art of profiling a serving workload is arranging for your profiler to start collecting *after* warmup and stop *before* the process winds down, while real batches are flowing.

This is not a stylistic preference; it is a correctness requirement, and it has a concrete mechanical basis. The GPU's SM clock is a function of load and temperature. On an idle A100 the SM clock can sit near its base frequency; under sustained load it boosts toward its maximum. Achieved FLOP/s scales linearly with clock, so a kernel measured in the first iteration (cold clocks) can report a throughput 20–30% below the same kernel measured at iteration 500 (boosted clocks). Nsight Compute, which we will meet at rung four, *locks the clocks* by default (`--clock-control=base`) precisely to make measurements reproducible — but that means an `ncu` number is a *base-clock* number, and you must know that when you compare it to a wall-clock measurement taken at boost. Being sloppy about the regime is how people produce profiling data that contradicts itself.

There is a second, subtler reason steady state matters for LLM serving specifically: **the batch composition is part of the workload**. A decode step with a batch of 4 sequences is a completely different kernel-shape distribution than a decode step with a batch of 96, and the arithmetic intensity of the projection matmuls scales with batch size. If you profile at batch size 1 (which is what happens if you send a single request to an idle server), you will measure the pathological memory-bound floor and conclude the whole system is memory-bound — which is true at batch 1 and progressively less true as continuous batching fills the GPU. Always profile under a batch that matches the load you are actually being asked to serve. Drive the server with a load generator (`vllm bench serve`, `genai-perf`, or a simple `asyncio` client fanning out N concurrent requests) and let the batch fill before you start collecting.

The practical recipe that falls out of all this: **warm up, then bracket a fixed window.** Every tool in this post has a mechanism to skip the first K iterations and capture a bounded region — `torch.profiler` has its `schedule`, `nsys` has `--capture-range` and `--delay`, and `ncu` has `-s`/`--launch-skip`. Use them. The single most common mistake I see in shared profiles is a timeline whose first 80% is cold start, with the reviewer earnestly discussing a `cudaMalloc` that happens once. Skip it.

How do you know you have reached steady state, rather than assuming it? Two cheap signals settle it. The first is the clock: poll `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -l 1` while the load generator ramps, and wait until the SM clock stops climbing and settles at its sustained boost frequency — that is the hardware telling you the boost ramp is finished and further measurements will not drift with clock. The second is the latency itself: watch the server's `vllm:time_per_output_token_seconds` histogram and wait until its p50 stops sliding and its p50-to-p99 spread narrows into a steady band. When both the clock and TPOT have flatlined, every iteration now resembles every other one, and the window you bracket will be representative rather than a slice of the ramp. A crude but effective automated gate is to require N consecutive decode steps whose per-step time is within a few percent of their running median before you call `torch.cuda.profiler.start()`:

```python
import time, statistics, collections

window = collections.deque(maxlen=50)          # last 50 decode-step times, in ms
while True:
    t0 = time.perf_counter()
    engine.step()                              # one decode iteration under real load
    window.append((time.perf_counter() - t0) * 1e3)
    if len(window) == window.maxlen:
        med = statistics.median(window)
        spread = (max(window) - min(window)) / med
        if spread < 0.05:                      # <5% jitter: clocks warm, caches full
            break                              # steady state reached; open the capture now
```

The threshold is deliberately loose — 5% jitter is achievable on a warm server and tight enough to exclude the ramp. On a noisy multi-tenant node you may never get under 5%, which is itself a finding: the interference is part of your production reality, and profiling *through* it (with the noise present) is more honest than waiting for a quiet window you will never see in production.

## The profiling ladder: only descend when you must

The reason the ladder in figure one is drawn as rungs, not a flat menu, is that each rung is dramatically more expensive to run and to read than the one above, and each answers a narrower question. Descending prematurely wastes hours; refusing to descend when the coarse metric is ambiguous wastes days. The discipline is: **let each rung tell you whether — and where — to descend.**

Rung one is your **server metrics**, and they are effectively free. Any production LLM server already exports them: vLLM ships a Prometheus `/metrics` endpoint out of the box with `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, and request/generation throughput. These cost nothing to collect (the server is emitting them regardless) and they answer the *first* question: is there even a problem, and is it a queueing problem or an execution problem? If `num_requests_waiting` is high and `gpu_cache_usage_perc` is pinned near 1.0, you are KV-cache-capacity-bound and the answer is more memory or better cache management, not a kernel profiler. If TTFT is fine but TPOT (time-per-output-token) is high while the queue is empty, the decode step itself is slow — *that* is when you descend. The companion post on [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving) covers building these dashboards properly; here they are simply the trigger that tells you whether kernel-level profiling is even the right tool.

Rung two is the **PyTorch profiler** (`torch.profiler`). It runs inside your Python process, hooks the CUDA runtime through CUPTI, and produces a per-operator breakdown plus a Chrome/Perfetto trace. Its overhead is modest — roughly 1.1–1.3× wall-clock with default settings — and it answers the *second* question: which operators and which host-side calls dominate, and is the GPU idle waiting on the CPU? It is the right rung when the server metric says "the decode step is slow" and you need to know *which part* of the step.

Rung three is **Nsight Systems** (`nsys`). It is a system-wide, timeline-oriented sampler: it captures CUDA API calls on the CPU threads, the kernels they launch on the GPU streams, memory copies, NCCL collectives, cuBLAS/cuDNN ranges, your NVTX annotations, and — with `--gpu-metrics-device` — a low-rate hardware-counter ribbon (SM active %, DRAM bandwidth) sampled across the whole run. Overhead is low (typically well under 1.2×) because it *samples and traces* rather than replaying. It answers the *third* question: on the wall-clock timeline, where are the gaps, is the CPU keeping the GPU fed, and is communication overlapped with compute?

Rung four is **Nsight Compute** (`ncu`). It is a per-kernel microscope, and it is expensive: to collect a full metric set it *replays each kernel many times*, serializing execution and inflating wall-clock by 100–1000×. You never run it across a whole server; you run it against one or a handful of named kernels that rung three has already fingered as the hot spot. It answers the *fourth and final* question: for this specific kernel, what fraction of peak DRAM bandwidth and peak compute is it reaching, what is its occupancy, and why are its warps stalling?

The ordering is not negotiable. Running `ncu --set full` against a whole vLLM server is a way to wait an hour for a report you cannot read. Running it against `rmsnorm_kernel` because the `nsys` timeline showed that kernel eating 12% of every decode step is fifteen minutes and one clear answer. The ladder exists so you pay rung four's price exactly once, on exactly the right kernel.

## Rung 2: torch.profiler — localize the hot region

The PyTorch profiler is the workhorse, because it is the cheapest tool that still shows you both sides of the CPU-GPU boundary. The key to using it on a serving workload is the `schedule`: you tell it to *wait* through warmup, *warm up* its own buffers, then collect a bounded number of *active* steps, so you capture steady state and not cold start. Here is a decode-step capture wired into a generation loop, exporting both a Chrome trace and a TensorBoard trace, with a `record_function` range marking the part I care about.

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, record_function, tensorboard_trace_handler

# schedule: skip 20 warmup steps, let the profiler warm its own buffers for 2,
# then actively record 5 steady-state steps, once.
prof_schedule = schedule(wait=20, warmup=2, active=5, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./tb_traces"),  # writes a TB-loadable trace
    record_shapes=True,      # tensor shapes per op — needed to spot shape-driven recompiles
    profile_memory=True,     # track allocator activity
    with_stack=True,         # Python stack for each op (higher overhead; drop if noisy)
) as prof:
    for step in range(64):                       # drive real decode steps
        with record_function("decode_step"):     # a named range you will see in the trace
            tokens = engine.step()               # one continuous-batching iteration
        prof.step()                              # advance the profiler's schedule

# Also emit a raw Chrome trace for chrome://tracing or ui.perfetto.dev
prof.export_chrome_trace("decode_trace.json")

# And the flat table, sorted by GPU time, for a quick text read
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="cuda_time_total", row_limit=25))
```

The `key_averages().table()` output is the fastest thing to read, and for a decode-bound LLM it has a very recognizable shape. You will see a long tail of tiny CUDA kernels — element-wise `mul`, `add`, `rms_norm`, `rotary_embedding`, the SwiGLU gate — each with small `CUDA total` but appearing thousands of times, and a couple of GEMMs (the QKV and MLP projections) and the attention kernel. The number that matters most is not any single op; it is the relationship between **`Self CPU total`** and **`Self CUDA total`** at the loop level. If the summed CPU time for launching the step's kernels is comparable to or larger than the GPU time those kernels run for, the GPU is starving — you are launch-bound, and the profiler is telling you the CPU cannot issue work fast enough. That is the signature that sends you to CUDA graphs, and it is visible right here at rung two before you ever open `nsys`.

The `record_function("decode_step")` range is what makes the Chrome trace readable. Without it, the Perfetto UI is an undifferentiated wall of ops; with it, you get a labeled band you can zoom into and measure. Add a handful of these around the phases you care about — `qkv_proj`, `attention`, `mlp`, `sampling` — and the trace becomes a map instead of a fog. This is also where you catch the classic serving-specific pathology: **CPU-side work that is not GPU work at all.** Token sampling, detokenization, logit processing (repetition penalties, JSON-schema-constrained decoding), and the scheduler's Python bookkeeping all run on the CPU between GPU steps. On a fast GPU with a slow sampler, the `sampling` range can be a bigger slice of TPOT than the entire transformer forward pass, and no amount of kernel optimization will touch it. The PyTorch profiler shows CPU and GPU on the same timeline, so this jumps out.

One caveat specific to serving frameworks: vLLM and TGI run the model in a worker process (sometimes a separate process per tensor-parallel rank), and the generation loop is buried inside the engine. You usually cannot wrap `engine.step()` as cleanly as the snippet above. In practice you either (a) use vLLM's built-in profiler hooks — vLLM's `LLM` and the OpenAI server honor the `VLLM_TORCH_PROFILER_DIR` environment variable, and hitting the `/start_profile` and `/stop_profile` endpoints (or calling `llm.start_profile()` / `llm.stop_profile()`) brackets a torch-profiler capture around live traffic — or (b) drop straight to `nsys`, which does not require you to instrument the loop at all. For a running server, option (a) is the cleanest way to get a torch trace of steady-state decode:

```bash
# Start the vLLM OpenAI server with the torch profiler enabled and pointed at a dir.
VLLM_TORCH_PROFILER_DIR=/traces/vllm \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 --max-num-seqs 64

# In another shell: warm up with load, THEN bracket a capture window.
curl -X POST http://localhost:8000/start_profile
#   ... let 5-10 seconds of representative traffic flow ...
curl -X POST http://localhost:8000/stop_profile
# A .pt.trace.json lands in /traces/vllm — open it in ui.perfetto.dev.
```

That gives you a real steady-state decode trace from the actual server, captured while a load generator hammers it, with no code changes to the engine. Read the trace, find the dominant band, and decide whether you even need to descend to `nsys`.

## Rung 3: Nsight Systems — read the timeline

When the torch trace says "the GPU is idle between kernels" or "attention dominates but I cannot see why," you descend to Nsight Systems. `nsys` is the tool that draws the *true* wall-clock timeline: every kernel as a colored bar on its stream, every `cudaLaunchKernel` on the CPU thread that issued it, every `cudaMemcpyAsync`, every NCCL collective, and your NVTX ranges as labeled spans. It is the single most valuable view in the whole ladder because it shows you *time you are not spending computing*, which is exactly what a launch-bound or comms-bound workload is made of.

The invocation for a serving workload has two parts: what to trace, and how to bracket the window. Here is the command I actually use against a vLLM server, gating the capture on the CUDA profiler API so it only records the window I explicitly open:

```bash
# Launch the server UNDER nsys, but do not start collecting yet: the capture range
# is gated on cudaProfilerStart/Stop, which we trigger from the client side.
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \   # CUDA kernels+API, NVTX ranges, OS runtime, GEMM/conv libs
  --gpu-metrics-device=all \              # low-rate HW counter ribbon: SM active %, DRAM BW
  --capture-range=cudaProfilerApi \       # only record between cudaProfilerStart and cudaProfilerStop
  --capture-range-end=stop \              # end collection when Stop is called
  --cuda-graph-trace=node \               # expand CUDA-graph nodes so replayed kernels are visible
  --sample=cpu \                          # periodic CPU stack samples (finds host hot spots)
  --output=/traces/vllm_decode \          # writes vllm_decode.nsys-rep
  vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --max-num-seqs 64
```

The trigger side is a one-liner in whatever drives the capture — vLLM's profiler hooks call `torch.cuda.profiler.start()` / `stop()` (which are `cudaProfilerStart` / `Stop` under the covers), so the same `/start_profile` / `/stop_profile` endpoints that bracket a torch trace also bracket the `nsys` capture range. If you are instrumenting your own loop, it is explicit:

```python
import torch
# ... warm up the server with representative load until the batch is full ...
torch.cuda.profiler.start()     # opens the nsys capture range (cudaProfilerStart)
for _ in range(30):             # ~30 steady-state decode steps is plenty to see the pattern
    engine.step()
torch.cuda.profiler.stop()      # closes it (cudaProfilerStop); nsys writes the report
```

A note on the flags. `--trace=cuda,nvtx,osrt` is the task's minimal recommendation and it is the right core: `cuda` for kernels and API calls, `nvtx` for your labeled ranges, `osrt` for OS-runtime blocking calls (so you can see when a CPU thread is stuck in a `futex` or a `poll` instead of launching kernels). I add `cublas`/`cudnn` so GEMMs are labeled with their library call, and `--gpu-metrics-device=all` so the timeline carries a hardware ribbon showing SM-active and DRAM-throughput percentages sampled across the run — that ribbon is a free, coarse version of the `ncu` numbers and often tells you memory-bound vs launch-bound without descending to rung four. `--cuda-graph-trace=node` matters specifically for LLM serving: if the framework replays a captured CUDA graph, by default `nsys` shows the whole graph as one opaque bar, and you want it expanded into its constituent kernels to see what is inside.

For **NVTX annotation** — the thing that turns a wall of kernels into a readable story — you push and pop named ranges around the phases of your step. In a serving codebase you add these once and they pay off on every future profile:

```python
import torch.cuda.nvtx as nvtx

def decode_step(model, batch):
    nvtx.range_push("decode_step")           # one span covering the whole step
    nvtx.range_push("qkv_proj")
    q, k, v = model.qkv(batch.hidden)        # the projection GEMM
    nvtx.range_pop()
    nvtx.range_push("attention")
    ctx = model.attn(q, k, v, batch.kv_cache) # the paged-attention kernel
    nvtx.range_pop()
    nvtx.range_push("mlp")
    out = model.mlp(ctx)                      # SwiGLU MLP
    nvtx.range_pop()
    nvtx.range_pop()
    return out
```

Now the `nsys` timeline has four nested, colored bands per step, and you can point at any gap and say exactly which phase it precedes. This is how you go from "the GPU has gaps" to "there is a 40 µs gap between the end of `attention` and the start of `mlp`, on the CPU thread, and it is a `futex` wait" — a claim precise enough to fix.

![A left-to-right Nsight Systems timeline of one steady-state decode step showing prefill GEMMs, a burst of tiny decode kernels, a CPU launch gap with the GPU idle, an un-overlapped NCCL all-reduce, and the next decode step waiting on the CPU](/imgs/blogs/profiling-llm-serving-with-nsight-2.webp)

The figure above is what that timeline looks like for a tensor-parallel decode step, laid out as the sequence you actually see in `nsys`. Prefill is a dense block of compute-bound GEMMs with the tensor cores genuinely busy (the ribbon shows high SM activity). Then decode step N is a *burst* of roughly 280 tiny kernels, each running for single-digit microseconds and each memory-bound. Between the kernels and before the next step there is a **CPU launch gap** — the GPU has finished the issued work and is sitting idle while the CPU thread catches up on launching the next batch of kernels. And because this is tensor-parallel, there is an **NCCL all-reduce** to sum the partial results across the two ranks, and if it is not overlapped with compute it is pure bubble: every rank waits at the collective. The sum of the gaps and the bubble is why TPOT is 24 ms instead of the ~16 ms the compute alone would justify. Reading this one picture is the whole point of rung three.

## Reading the timeline: the four classic problems

Once you can read an `nsys` timeline, four failure signatures recur across nearly every LLM serving workload, and each has a distinct fix. Learning to recognize them by shape is most of the value of profiling, because the shape *is* the diagnosis.

The first is **launch-bound decode**: the timeline shows hundreds of tiny kernels separated by visible gaps on the GPU stream, and the CPU thread above them is a solid wall of `cudaLaunchKernel` calls with no idle time. The GPU is waiting on the CPU. This happens because a transformer decoder layer decomposes into dozens of small operations (norms, rotary embeddings, element-wise gates, residual adds, small GEMVs), and at batch sizes where each kernel runs for only a few microseconds, the fixed ~5–10 µs cost of *launching* a kernel from the CPU becomes the dominant term. The fix is not a faster kernel — the kernels are already fast — it is to stop launching so many of them, which is exactly what CUDA graphs do: capture the entire sequence of launches once, then replay it as a single GPU-side submission that needs no per-kernel CPU involvement.

![A before-and-after contrast of an eager decode step with 300-plus kernel launches and 35 percent GPU idle versus a CUDA-graph replay with one launch, gaps eliminated, and GPU idle under 5 percent](/imgs/blogs/profiling-llm-serving-with-nsight-3.webp)

The figure above shows the before-and-after in the terms you read off the timeline. On the left, eager execution: 300-plus launches at ~8 µs of CPU each, CPU-side gaps where the GPU waits, and the GPU idle 35% of the step with TPOT at 24 ms. On the right, the same decode step captured as a CUDA graph and replayed: one launch, the CPU off the critical path entirely, GPU idle under 5%, and TPOT down to 16 ms. The [kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) post covers the capture mechanics and their sharp edges (fixed shapes, no data-dependent control flow); the point here is that the *profile* is what tells you the fix is worth doing. If your decode step were already compute-bound, capturing it as a graph would buy nothing.

#### Worked example: reading an nsys trace to find the launch-bound gap

Concretely, here is how you confirm launch-bound from a real trace rather than eyeballing it. Take Llama-3.1-8B on a single A100 80GB, batch size 32, decode phase, captured with the command above. Open `vllm_decode.nsys-rep` in the Nsight Systems GUI (or run `nsys stats` on it, below) and measure three things.

First, the **GPU-idle fraction within a step**. Zoom to one `decode_step` NVTX span. Its wall-clock width is, say, 24 ms. Now select the GPU stream row and sum the kernel durations inside the span — suppose it totals 15.6 ms of actual kernel execution. The GPU was idle `24 - 15.6 = 8.4 ms`, or 35% of the step. That idle time is the money.

Second, **where the idle sits**. Zoom into a gap between two kernels. If the CPU thread directly above shows a `cudaLaunchKernel` (or `cudaStreamSynchronize`) spanning the gap, the GPU is waiting on the CPU to issue the next kernel — launch-bound. If instead the gap aligns with a `cudaMemcpyAsync` or an NCCL span, it is a data-movement or comms bubble, a different problem.

Third, **the kernel count and mean duration**. Run the stats report from the command line so you have numbers, not impressions:

```bash
# Summarize kernel occupancy of GPU time and per-kernel counts from the report.
nsys stats --report cuda_gpu_kern_sum vllm_decode.nsys-rep | head -30
```

```console
 Time(%)  Total Time (ns)   Instances   Avg (ns)   Med (ns)   Name
 -------  ---------------  -----------  ---------  ---------  ----------------------------------
    18.4       2,870,400        9,600       299       288    void vllm::rms_norm_kernel<...>
    15.1       2,355,000        4,800       490       472    void vllm::rotary_embedding_kernel
    12.7       1,981,000        4,800       412       400    void vllm::act_and_mul_kernel<...>   # SwiGLU
    11.9       1,856,000        1,600     1,160     1,120    ampere_fp16_s16816gemm_...           # QKV/MLP GEMM
    10.2       1,591,000          800     1,988     1,940    void flash_fwd_kernel<...>           # attention
     ...
```

Nine thousand six hundred instances of `rms_norm_kernel`, each averaging 299 ns, across 30 steps — that is 320 launches per step for that one kernel family, each running for less time than the ~5 µs it costs the CPU to launch it. When the *median kernel duration is smaller than the launch cost*, you are launch-bound by definition, and the kernel-summary report proves it with counts. The fix (CUDA graphs) collapses those launches; you re-run the exact same measurement afterward and confirm the idle fraction dropped below 5%. Measure, fix, re-measure — never fix and hope.

The second signature is a **memory-bound kernel**: one kernel dominates the step and the GPU-metrics ribbon shows DRAM throughput pinned near peak while SM (compute) throughput stays low. This is not a bug — a decode GEMV or an RMSNorm *should* be memory-bound, because its arithmetic intensity is far below the roofline ridge. The fix is never "tune the kernel's occupancy"; it is "move less data," which means fusion (do more work per byte read from HBM) or quantization (make each weight fewer bytes). We take this one to rung four below, because confirming it needs the per-kernel DRAM percentage that only `ncu` measures precisely.

The third signature is **low occupancy**: a kernel runs for a long time but the ribbon shows *both* DRAM and compute throughput low. The GPU is neither memory- nor compute-saturated — it is stalling, usually because too few warps are resident to hide latency (a launch configuration with too few blocks, or a kernel using so many registers per thread that few warps fit per SM). This is the case where tuning the block size or the register budget actually helps, and it is the *only* one of the four where "make the kernel itself better" is the right move.

The fourth signature is an **NCCL bubble**: in a tensor-parallel or pipeline-parallel deployment, the timeline shows an NCCL `all_reduce` (TP) or `send`/`recv` (PP) span during which the compute stream is idle. Communication is not overlapped with computation, so every rank pays the full collective latency as dead time. The fix is to overlap comms with compute — issue the collective on a separate CUDA stream and structure the computation so useful work happens while the bytes are in flight — or, at the layer of the framework, to use a communication-overlapping implementation. The [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) post covers the parallelism strategies whose bubbles you are looking at here.

#### Worked example: sizing an NCCL comms bubble on a TP timeline

The fourth signature is the one people most often assume rather than measure, so here is how to put a number on it instead of a hunch. Run the server at `--tensor-parallel-size 2`, capture with NCCL tracing added, and read the collective time straight off the report rather than eyeballing the ribbon:

```bash
# Trace NCCL collectives alongside CUDA kernels and NVTX, bracketed to steady state.
nsys profile --trace=cuda,nvtx,nccl \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/traces/vllm_tp2 \
  vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --max-num-seqs 64

# NCCL collectives execute as CUDA kernels; sum their GPU time from the kernel report.
nsys stats --report cuda_gpu_kern_sum /traces/vllm_tp2.nsys-rep | grep -iE 'nccl|allreduce'
```

Suppose the all-reduce kernels total 3.1 ms of GPU time across a 24 ms window of decode steps. That number alone is not the bubble — the question is how much of it overlaps compute. Zoom to one decode step in the GUI, put the compute stream and the NCCL stream side by side, and measure the span where the compute stream is idle *while* the collective runs. If the all-reduce is 130 µs per layer and the compute stream sits idle for 110 µs of it, then 85% of the collective is bubble rather than overlap — and across 32 layers that is roughly 3.5 ms of a 24 ms step spent doing nothing but waiting on NVLink. Now the fix has a target: move the all-reduce onto a side stream so the next layer's input projection issues while the bytes are in flight, and re-measure the idle span. If overlap works, the same grep shows similar total collective time but the per-step idle drops, and TPOT falls by the recovered bubble. And if dropping from TP=2 to TP=1 fits the model in a single card, the collective vanishes entirely — a larger win than any overlap, and one you only trust because you sized the bubble first instead of guessing it was the problem.

## Rung 4: Nsight Compute — one kernel under the microscope

When rung three has named the guilty kernel, you descend to Nsight Compute for the definitive answer on *that kernel*. `ncu` is different in kind from `nsys`: it does not sample a timeline, it *profiles a kernel exhaustively* by replaying it — running it once per metric group it needs to collect, with the GPU in a controlled state — and reporting hundreds of hardware counters. That replay is why it is 100–1000× slower and why you never point it at a whole server. You point it at one kernel, a handful of times, and read a report card.

The invocation filters to the kernel of interest and skips warmup so you profile a steady-state instance:

```bash
# Profile the attention kernel specifically. -k is a regex on the kernel name;
# -s skips the first 200 launches (warmup), -c collects the next 3 instances.
ncu \
  --set full \                              # collect all sections (Speed-of-Light, occupancy, stalls, ...)
  --kernel-name "regex:flash_fwd|paged_attention" \
  --launch-skip 200 \                       # skip warmup launches
  --launch-count 3 \                        # profile 3 steady-state instances
  --clock-control base \                    # lock clocks for reproducibility (default)
  --export /reports/attn_report \           # writes attn_report.ncu-rep (open in the ncu UI)
  python drive_one_request.py               # a tiny script that runs decode under load
```

Two practical notes. First, `--set full` is thorough but slow; for a quick memory-vs-compute verdict, `--section SpeedOfLight --section Occupancy` collects just the Speed-of-Light throughput bars and the occupancy numbers in a fraction of the time. Second, because `ncu` serializes and replays, you cannot run it against the live production server taking real traffic — you run it against a small harness (`drive_one_request.py`) that reproduces the same kernel shapes (same batch, same sequence length, same dtype) in isolation. Matching the shapes to production is the whole trick: the same kernel at batch 4 versus batch 96 gives a completely different report.

When you want a specific number in a script rather than the full report, `ncu` can print named metrics directly to CSV — this is how you build the parsing tool in the triage section:

```bash
# Pull just the four numbers that classify a kernel, as CSV, for one instance.
ncu --metrics \
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name "regex:rms_norm" --launch-skip 200 --launch-count 1 \
  --csv --page raw python drive_one_request.py
```

Those four metric identifiers are the ones worth memorizing: `gpu__dram_throughput...pct_of_peak` is the **Memory (DRAM) Throughput** — the fraction of peak HBM bandwidth the kernel sustains; `sm__throughput...pct_of_peak` is the **Compute (SM) Throughput**; `sm__pipe_tensor_op_hmma_cycles_active...` is the **tensor-core active** percentage (how busy the matrix-multiply pipeline is); and `sm__warps_active...pct_of_peak` is the **achieved occupancy**. Four numbers, and between them they place any kernel into launch-, memory-, or compute-bound.

![A six-cell Nsight Compute report card for a decode RMSNorm kernel showing 41 microsecond duration, 89 percent DRAM throughput, 62 percent occupancy against a 66 percent theoretical maximum, zero tensor-core activity, and a memory-bound verdict recommending fusion](/imgs/blogs/profiling-llm-serving-with-nsight-4.webp)

The figure above is a real report card for the RMSNorm kernel that `nsys` fingered — read it as a verdict, not a data dump. Duration 41 µs, 12% of the decode step. DRAM throughput 89% of peak: the kernel is saturating HBM bandwidth. SM occupancy 62% against a theoretical maximum of 66%, so it is *not* occupancy-limited — there are plenty of warps resident. Tensor-core active 0%: there is no matrix multiply here at all, it is pure element-wise arithmetic. The pattern — near-peak memory throughput, zero tensor activity, healthy occupancy — is the unambiguous signature of a **memory-bound kernel operating near its floor**. The kernel is already about as fast as it can be given how many bytes it must move; the only way to make it faster is to make it move fewer bytes. That is a fusion problem, not a kernel-tuning problem, and the report card told you so in four numbers.

## The metrics that matter, and what they actually mean

Two of those numbers — **SM occupancy** and **DRAM throughput %** — are the ones people most often misread, so it is worth being precise about the mechanics, because the whole diagnosis rests on interpreting them correctly.

**SM occupancy** is the ratio of *active warps resident on a streaming multiprocessor* to the *maximum warps that SM can hold*. A warp is 32 threads executed in lockstep; an SM on an A100 can hold up to 64 warps (2048 threads). Occupancy is a **latency-hiding** metric, and that is the key to reading it. When a warp issues a memory load, it stalls for hundreds of cycles waiting for HBM. The SM hides that stall by switching to another resident warp that is ready to issue. The more warps resident (higher occupancy), the more independent work is available to cover memory latency. So occupancy answers one question: *does this kernel have enough parallelism resident to hide its own memory stalls?*

The crucial subtlety — the one that trips people up — is that **high occupancy does not imply high performance, and 100% occupancy is rarely the goal.** A memory-bound kernel at 60% occupancy can already be saturating DRAM bandwidth; pushing occupancy to 90% moves no additional bytes and changes nothing, because the bottleneck is the memory bus, not the amount of resident work. Conversely, a compute-bound kernel can hit peak FLOP/s at 50% occupancy if its instructions are independent enough to keep the pipelines full (this is the classic result behind FlashAttention-2's design — high throughput at moderate occupancy). Occupancy only matters when it is *low enough to be the bottleneck*: below roughly 25–30% you usually cannot hide memory latency and the kernel stalls waiting on loads. That is why the triage rule is "low occupancy AND low DRAM AND low compute → fix the launch config"; occupancy in isolation is not actionable.

One more distinction sharpens occupancy into something you can act on: **theoretical versus achieved**. Theoretical occupancy is the ceiling fixed by the kernel's launch configuration — its registers per thread, its shared-memory request per block, and its block size — computed before the kernel ever runs; `ncu` reports it in the Occupancy section together with which of those three resources is the binding limiter. Achieved occupancy is what the kernel actually sustained at runtime, always at or below theoretical, and it falls further when the work is unbalanced — a tail of straggler blocks, or a grid too small to fill every SM. The gap between the two numbers is itself the diagnosis. If theoretical is 50% but achieved is only 20%, the launch *could* hold more warps than the runtime kept resident, which points at grid sizing or load balance rather than a register ceiling — you need more blocks, or a more even distribution of work across them. If theoretical is *itself* capped at 25% and `ncu` names registers as the limiter, no amount of grid tuning helps; the fix is the register budget. Reading the two together tells you whether to change how the kernel is launched or how it is compiled — a level of specificity a single occupancy percentage can never give you.

**DRAM throughput %** is the fraction of the GPU's peak HBM bandwidth the kernel sustains, averaged over its runtime. This one is more directly interpretable: it is a measure of *how close the kernel is to the memory roofline*. A kernel at 85–95% DRAM throughput is moving bytes about as fast as the hardware physically can — it is memory-bound and near its floor. The important inference is what it means for the fix: if DRAM% is near peak, the kernel is *not slow because it is badly written*; it is slow because it must move that many bytes. Reducing its time requires reducing its byte traffic — fuse it with a neighbor so an intermediate stays in registers/SRAM instead of round-tripping through HBM, or quantize the data it reads so each element is fewer bytes. Trying to "optimize" a 90%-DRAM kernel by tuning its block size is wasted effort; the hardware is already delivering its bandwidth.

The two numbers read together give you the diagnosis directly. High DRAM% and low compute% is memory-bound. High compute% (and high tensor-active% for a GEMM) is compute-bound. *Both* low means the kernel is latency-bound — stalling — and then you look at the **warp-stall reason** to learn why: in Nsight Compute's Warp State Statistics, a dominant "Stall Long Scoreboard" means warps are waiting on global-memory loads (an L2/DRAM latency problem, often fixable with better memory access patterns or more occupancy), while "Stall MIO Throttle" or "Stall Math Pipe Throttle" point at instruction-issue bottlenecks. The stall reason is the tiebreaker when the throughput bars are ambiguous.

![A five-row-by-three-column matrix mapping SM occupancy, DRAM throughput percent, tensor-core active percent, achieved FLOP per second, and warp stall reason to what each measures, its healthy range in decode, and the bound it signals](/imgs/blogs/profiling-llm-serving-with-nsight-5.webp)

The figure above collapses this into a decoder ring: each counter, what it measures, what a healthy decode value looks like, and — the actionable column — the bound it signals. Note the deliberate asymmetry in the "healthy in decode" column: a decode step *wants* high DRAM throughput (it is supposed to be memory-bound) and *low* tensor-core activity (there is little matmul at batch sizes typical of latency-sensitive serving), while a prefill step is the opposite. "Healthy" is workload-relative, which is why the same counter reads as good news in one phase and a red flag in another. A decode kernel at 5% DRAM and 0% compute is a launch-bound stall; a prefill GEMM at 5% DRAM and 0% tensor-core would be badly broken.

## Mapping a kernel to the roofline

The single most useful mental frame for turning these counters into a verdict is the roofline model, and Nsight Compute will even draw it for you — the `--set full` report includes a roofline chart in the GPU Speed of Light section, plotting the kernel's achieved performance against its arithmetic intensity, with the memory-bandwidth ceiling (the sloped line) and the compute ceiling (the flat line) drawn in. Where the kernel's dot lands tells you everything: under the sloped line means memory-bound (you are limited by bandwidth), under the flat line means compute-bound.

The mechanics are worth stating precisely because they connect directly to the counters. **Arithmetic intensity** (AI) is FLOPs performed divided by bytes moved to and from HBM. The **ridge point** — the AI at which a kernel transitions from memory- to compute-bound — is the ratio of the GPU's peak compute to its peak bandwidth:

$$
\text{ridge} = \frac{\text{peak FLOP/s}}{\text{peak bytes/s}}.
$$

For an A100 80GB that is about $312 \times 10^{12} / 2.0 \times 10^{12} \approx 156$ FLOP/byte; for an H100 SXM about $990 \times 10^{12} / 3.35 \times 10^{12} \approx 295$ FLOP/byte. Any kernel whose AI is below the ridge is memory-bound and its wall-clock time is $\text{bytes} / \text{bandwidth}$; above the ridge it is compute-bound and time is $\text{FLOPs} / \text{peak}$. A decode projection at batch size 1 reads a weight matrix once per token and does two FLOPs per weight element, giving AI ${\approx 1}$ FLOP/byte — more than a hundred times below the A100 ridge, which is why decode is memory-bound to the point of being a memory copy with a multiply stapled on. The full derivation of the LLM roofline, and how batch size slides you up the intensity axis, lives in the sibling post on [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference); here the point is operational: the DRAM% and compute% counters from `ncu` are literally the coordinates of the kernel's dot on that chart.

#### Worked example: reading an ncu report to fix a memory-bound kernel

Return to the RMSNorm report card. The kernel reads its input activations from HBM, computes a normalization, and writes the result back — for a hidden size of 4096 and batch 32 in FP16, that is roughly $32 \times 4096 \times 2$ bytes in plus the same out, about 0.5 MB of traffic, and a few FLOPs per element. Its AI is well under 1, so on the roofline it sits far under the sloped memory line — memory-bound, exactly as the 89% DRAM throughput confirmed. The kernel is at its floor: $0.5\ \text{MB} / 2.0\ \text{TB/s} \approx 0.25\ \mu s$ of pure bandwidth time, and with launch and setup overhead it measures 41 µs of a step that has hundreds of such kernels.

You cannot make this kernel move fewer *of its own* bytes — it must read the activations and write them. But you *can* eliminate the round-trip by **fusing** it: RMSNorm's output feeds directly into the QKV projection GEMM, so if you fuse the normalization into the GEMM's prologue (or fuse the residual-add + norm into one kernel), the normalized activations never get written to HBM and re-read — they stay in registers or shared memory and flow straight into the matmul. That removes one full read and one full write of the activation tensor from the step's HBM traffic. Do this across all the element-wise ops (norm, rotary, gate) and you cut both the *kernel count* (helping the launch-bound problem from earlier) and the *byte traffic* (helping the memory-bound problem). The [custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference) post shows how to write such a fused kernel in Triton and wire it into vLLM; the profiler is what told you the fusion was worth writing — because the report card proved the kernel was memory-bound and near its floor, not compute-bound or occupancy-limited.

The discipline this worked example illustrates is the entire thesis: the *same symptom* ("this kernel is slow") has three different fixes depending on which bound the counters reveal, and guessing wrong wastes the effort. A slow kernel that is compute-bound wants a better algorithm or lower precision; one that is memory-bound wants fusion or quantization; one that is occupancy-limited wants a better launch configuration. Only the profile distinguishes them.

#### Worked example: reading an ncu report for a latency-bound kernel

Not every slow kernel is memory- or compute-bound; the third signature — both throughputs low — is the one where reading the report actually changes what you do, so it earns its own walk-through. Suppose `nsys` fingers a custom sampling routine or a small fused kernel that runs 60 µs, but whose GPU-metrics ribbon showed neither DRAM nor SM anywhere near peak. Profile it with only the sections that diagnose stalls, which is far cheaper than `--set full`:

```bash
# Occupancy + launch config + warp-stall breakdown, without a full metric replay.
ncu --section Occupancy --section LaunchStats --section WarpStateStats \
  --kernel-name "regex:sampling|topk" --launch-skip 200 --launch-count 2 \
  --export /reports/sampling_report python drive_one_request.py
```

Read three things off the report in order. First, **Achieved Occupancy** against **Theoretical Occupancy**: if achieved is 18% versus a theoretical 50%, the kernel is not filling the SMs it is allowed to — a runtime or launch problem, not a hardware ceiling. Second, the **limiter** named in the Occupancy section: `ncu` states whether registers, shared memory, or block size caps the theoretical number. If it reads `launch__occupancy_limit_registers`, the kernel is spending too many registers per thread (say 168), so few warps fit per SM; a `__launch_bounds__` hint or a `maxrregcount` cap trades a little spilling for more resident warps. Third, the dominant **warp-stall reason** in Warp State Statistics: `Stall Long Scoreboard` means warps are parked on global-memory loads (raise occupancy or fix the access pattern so more loads are in flight at once), while `Stall Not Selected` alongside low occupancy means there simply are not enough warps to issue from. Here — unlike the memory-bound case — the right move really is to tune the launch: raise the block count so more blocks are resident, or cut the register footprint so more warps fit. Re-run the identical three-section capture and confirm that achieved occupancy climbed and the dominant stall shifted; if the kernel now runs 38 µs, the launch configuration was the bottleneck all along. This is the one branch of the four where making the kernel itself better is the correct answer, and the report is what licenses that work instead of leaving it a guess.

## Bottleneck triage: symptom to metric to fix

Putting the four signatures and the counters together yields a triage procedure you can run mechanically. The tree below is the decision procedure: read the profile, see which of four things dominates, and each maps to exactly one class of fix.

![A four-way triage tree rooted at reading the profile, branching to big CPU gaps, one hot kernel at high DRAM percent, low SM occupancy, and an un-overlapped NCCL bubble, each leading to a distinct fix](/imgs/blogs/profiling-llm-serving-with-nsight-6.webp)

The tree in the figure above is the whole method in one picture: big CPU gaps with the GPU idle above 20% → CUDA graphs plus chunked prefill; one kernel hot at DRAM% above 80% → fuse or quantize to cut HBM traffic; SM occupancy below 30% with both throughputs low → tune the block size to raise occupancy; an NCCL bubble not overlapped with compute → move the collective to a separate stream and overlap it. Four branches, four fixes, and the profile picks the branch — you never try all four in turn.

It is worth writing this as a reference table too, because in practice you are matching an observed symptom against a fix and it helps to have the full mapping in one place:

| Symptom in the trace | The diagnostic metric | What it means | The fix |
|---|---|---|---|
| Tiny kernels, visible GPU-stream gaps, CPU wall of launches | GPU-idle fraction > 20%; median kernel dur < launch cost | Launch-bound: CPU cannot feed the GPU | CUDA graphs; fuse element-wise ops; reduce op count |
| One kernel dominates; ribbon shows DRAM pinned | `dram_throughput.pct_of_peak` > 80%, compute low | Memory-bound near floor | Fuse to cut round-trips; quantize weights/KV |
| Long kernel, both DRAM% and compute% low | `warps_active.pct_of_peak` < 30%; "Stall Long Scoreboard" | Occupancy/latency-bound | Tune block size, registers, shared-mem budget |
| GEMM slow, tensor cores under-used | `sm__pipe_tensor_op...active` low on a matmul | Not using tensor cores well | Fix dtype/alignment; use FA-2/cuBLASLt; check tile shape |
| NCCL span with idle compute stream | comms span not overlapping kernels | Comms bubble in TP/PP | Overlap collective on separate stream; tune TP degree |
| Decode-step width spikes periodically | wide steps coincide with prefill GEMMs on the stream | Prefill-decode interference | Chunked prefill; cap `--max-num-batched-tokens` |
| CPU range (sampling/detok) rivals forward pass | `torch.profiler` Self CPU on non-GEMM ranges | Host-side bottleneck | Faster sampler; async detokenize; move logit-proc off critical path |

The last row deserves emphasis because it is the one kernel-level profiling can *miss* if you only look at the GPU. On a fast enough GPU, the CPU-side work — sampling, detokenization, structured-output constraint checking, the scheduler's Python — can dominate TPOT while every GPU counter looks healthy. That is why py-spy belongs in the toolkit: it samples the Python call stack of a *running* process with no instrumentation and no restart, so you can see where the host is spending time between kernel launches.

```bash
# Attach to the live vLLM worker process and record a flamegraph of the Python side.
py-spy record --pid $(pgrep -f 'vllm.*EngineCore' | head -1) \
  --duration 30 --rate 250 --output vllm_cpu_flame.svg
# Or a one-shot snapshot of what every thread is doing right now:
py-spy dump --pid $(pgrep -f 'vllm.*EngineCore' | head -1)
```

Finally, the piece that makes triage repeatable across a team: **a script that turns a raw profiler dump into a one-line verdict.** Rather than eyeballing the `nsys stats` output every time, parse it and classify. This takes the `cuda_gpu_kern_sum` and `cuda_gpu_trace` reports `nsys` can emit as CSV and computes the launch-bound and memory-bound signals directly:

```python
#!/usr/bin/env python3
"""Classify an nsys report into launch- vs memory- vs compute-bound.
Usage: nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
                  --output . vllm_decode.nsys-rep   # writes *_cuda_gpu_kern_sum.csv etc.
       python classify_profile.py vllm_decode
"""
import csv, sys

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

base = sys.argv[1]
kern = load(f"{base}_cuda_gpu_kern_sum.csv")     # per-kernel GPU time + instance counts
api  = load(f"{base}_cuda_api_sum.csv")          # per-API-call CPU time (cudaLaunchKernel, ...)

# Total GPU kernel time and total launch-API time over the captured window.
gpu_ns    = sum(int(r["Total Time (ns)"]) for r in kern)
launches  = sum(int(r["Instances"]) for r in kern)
launch_ns = sum(int(r["Total Time (ns)"]) for r in api
                if r["Name"].startswith("cudaLaunchKernel"))
mean_kern_ns = gpu_ns / max(launches, 1)

# Heuristic classification. Launch cost per kernel is ~5000 ns on a typical host.
LAUNCH_COST_NS = 5000
verdict = []
if launch_ns > 0.5 * gpu_ns or mean_kern_ns < LAUNCH_COST_NS:
    verdict.append(f"LAUNCH-BOUND (mean kernel {mean_kern_ns:.0f} ns < "
                   f"{LAUNCH_COST_NS} ns launch; {launches} launches) -> CUDA graphs / fusion")

# The single hottest kernel, as a share of GPU time — candidate for ncu drill-down.
hottest = max(kern, key=lambda r: int(r["Total Time (ns)"]))
share = int(hottest["Total Time (ns)"]) / max(gpu_ns, 1)
if share > 0.15:
    verdict.append(f"HOT KERNEL: {hottest['Name'][:48]} = {share:.0%} of GPU time "
                   f"-> profile with: ncu -k 'regex:{hottest['Name'].split('<')[0][-20:]}'")

print("\n".join(verdict) or "No dominant launch/kernel signal; check GPU-metrics ribbon and NCCL.")
```

Run it after every capture and it emits a verdict like `LAUNCH-BOUND (mean kernel 412 ns < 5000 ns launch; 9600 launches) -> CUDA graphs / fusion`, plus the exact `ncu` command to drill into the hottest kernel. That turns profiling from an artisanal skill into a checked-in tool the whole team can run, and it enforces the ladder: it will not send you to `ncu` unless one kernel actually dominates.

## The repeatable profiling SOP

The workflow that all of this composes into is worth stating as a standard operating procedure, because the value of profiling is destroyed by doing it ad hoc. The steps below are the loop you run every time, and the discipline is to never skip a rung and never fix without re-measuring.

![A branching workflow graph from coarse metrics to torch.profiler, splitting to nsys for the timeline and ncu for a single kernel, both merging into a roofline classification before a fix-and-re-measure step](/imgs/blogs/profiling-llm-serving-with-nsight-7.webp)

The graph above is the loop. Start at coarse metrics: is TTFT or TPOT actually breaching, and is it a queue problem or an execution problem? If execution, run `torch.profiler` to localize the hot region and decide which lower tool you need. Branch to `nsys` when the question is about *time between kernels* — gaps, overlap, NCCL — and to `ncu` when the question is about *one kernel's efficiency*. Both feed the roofline classification: memory-, compute-, or launch-bound. Only then apply the fix, and — the step people skip — **re-measure with the identical capture** to confirm the number moved. The re-measure is not optional bureaucracy; it is how you catch the depressingly common case where the "fix" helped the microbenchmark but regressed the end-to-end TPOT because it shifted the bottleneck somewhere else.

Written as a checklist, the SOP is:

1. **Confirm the problem** at rung one. If the queue is the bottleneck, this is not a kernel problem — stop and fix capacity or scheduling.
2. **Warm up and bracket.** Drive representative load, skip warmup, capture a bounded steady-state window. Never profile cold start.
3. **Localize** with `torch.profiler`: which phase dominates, and is the CPU keeping the GPU fed?
4. **Time it** with `nsys` if the issue is gaps/overlap/comms; annotate with NVTX so the timeline is readable. Run the classify script.
5. **Zoom** with `ncu` on the one kernel `nsys` fingered; read the four counters and the roofline dot.
6. **Fix the identified bound** — and only that bound.
7. **Re-measure** with the same capture. Confirm the target metric moved and TPOT/throughput improved end-to-end.
8. **Write down** the before/after numbers and the flag/change, so the next person does not re-derive it.

This is also where profiling connects to the broader debugging story: when the problem is correctness or a distributed hang rather than raw speed, the [tracing and debugging LLM serving](/blog/machine-learning/model-serving/tracing-and-debugging-llm-serving) post covers request-level tracing and the distributed-hang playbook that `nsys`'s multi-process timeline complements.

## Targets per GPU, and a measurement to anchor them

A number in isolation means nothing; "89% DRAM throughput" is only a verdict because you know the peak. The targets that make counters interpretable are fixed by the hardware, and they differ enough across cards that the *same kernel* is memory-bound on one GPU and nearly balanced on another.

![A four-row-by-four-column matrix of A100 80GB, H100 SXM5, L4, and H200 GPUs with their HBM bandwidth, peak dense BF16 throughput, roofline ridge point, and single-stream 8B decode floor](/imgs/blogs/profiling-llm-serving-with-nsight-8.webp)

The figure above tabulates the anchors. HBM bandwidth sets the memory roofline; peak dense BF16 sets the compute roofline; their ratio is the ridge point; and the single-stream decode floor — the minimum time to read an 8B model's 16 GB of FP16 weights once — is the concrete latency consequence. An A100 at 2.0 TB/s cannot decode 8B faster than ${16}\text{ GB} / 2.0\text{ TB/s} \approx 8.0$ ms per token single-stream (~125 tok/s); an H100 at 3.35 TB/s hits ~4.8 ms (~208 tok/s); an H200 at 4.8 TB/s reaches ~3.3 ms (~300 tok/s); an L4 at 0.30 TB/s is stuck at ~53 ms (~19 tok/s). These floors are *bandwidth* floors — no kernel cleverness beats them at batch 1 — and they are exactly why batching (which amortizes the weight read across many tokens) is the primary throughput lever for decode.

The healthy-range targets follow from the same table. For a **decode** kernel on any of these cards, healthy means high DRAM throughput (70–95% of that card's peak) and near-zero tensor-core activity — it is supposed to be bandwidth-bound. For a **prefill** GEMM, healthy means high tensor-core activity (50–75% on A100/H100 for a well-tuned FA-2 or cuBLASLt GEMM) and high compute throughput. Occupancy above ~50% is comfortable for decode; below ~30% is a red flag worth investigating. And the launch-bound tell is card-relative in one respect: the faster the GPU, the *shorter* the kernels run, so the fixed CPU launch cost becomes a larger fraction — which is why launch-bound decode is often *worse* on an H100 than an A100 for the same small model, and why CUDA graphs matter more as GPUs get faster.

To make those targets usable at the counter level, here is the same guidance as concrete bands you can check a report against. The values are the healthy ranges for a well-tuned server on a data-center card (A100/H100/H200 class); the L4 and other bandwidth-starved cards hit the same *shapes* at lower absolute clocks, so read the ranges as ratios-to-peak — which is exactly what the `pct_of_peak` metrics already report.

| Counter (`ncu` metric) | Decode (memory-bound) | Prefill (compute-bound) | Red flag |
|---|---|---|---|
| DRAM throughput (`gpu__dram_throughput.pct_of_peak`) | 70–95% | 20–50% | <30% in decode → launch/latency-bound |
| SM throughput (`sm__throughput.pct_of_peak`) | 20–50% | 60–90% | both DRAM and SM <30% → stalling |
| Tensor-core active (`sm__pipe_tensor_op_hmma...`) | 0–10% | 50–75% | <30% on a prefill GEMM → wrong dtype/tile |
| Achieved occupancy (`sm__warps_active.pct_of_peak`) | 50–75% | 40–70% | <30% → cannot hide latency |
| GPU-idle fraction within a step (nsys) | <5% (graphed) | <5% | >20% → launch-bound, add CUDA graphs |

The asymmetry between the decode and prefill columns is the whole point: a report is only healthy or unhealthy relative to which phase produced it. A 30% DRAM reading is a red flag for a decode kernel and completely normal for a compute-bound prefill GEMM, so always label a captured report with the phase it came from before you judge its numbers — the same counter value is a pass in one column and a fail in the other.

A concrete before/after measurement anchors all of this. Below is a representative profiling-driven optimization pass on Llama-3.1-8B, single node, batch 32 decode, measured before and after applying the fix the profile identified — the numbers are illustrative of the pattern and the ratios are what generalize, not the exact absolute values on your hardware:

| Fix (identified by the profile) | Hardware | Metric that moved | Before | After |
|---|---|---|---|---|
| CUDA graphs for launch-bound decode | A100 80GB, bs=32 | GPU-idle fraction / TPOT | 35% / 24 ms | <5% / 16 ms |
| CUDA graphs for launch-bound decode | H100 SXM5, bs=32 | GPU-idle fraction / TPOT | 42% / 15 ms | <5% / 9 ms |
| Fuse RMSNorm+residual into GEMM prologue | A100 80GB, bs=32 | Kernel count / step time | ~320 / 24 ms | ~200 / 22 ms |
| FA-2 attention vs unfused prefill attention | A100 80GB, prefill | Tensor-core active % | ~25% | ~60% |

Two things to read off this. First, the launch-bound fix (CUDA graphs) is a *larger* relative win on the H100 than the A100 — 42%→<5% idle versus 35%→<5% — exactly as the "faster GPU, shorter kernels, bigger launch fraction" argument predicts. Second, the fusion fix moves the kernel count more than the step time, because once you are already in a CUDA graph the launch cost is mostly gone and the remaining benefit is the HBM-traffic reduction — which is real but smaller than the launch win. The profile told you to do the CUDA graph *first* because that is where the idle time was; doing the fusion first would have delivered the smaller number and left the big one on the table. Order matters, and the profile sets the order.

## Case studies

**vLLM and CUDA graphs.** The launch-bound decode pattern is not hypothetical — it is the reason vLLM captures CUDA graphs by default and exposes `enforce_eager` to turn them off. In eager mode, every decode step re-issues the full sequence of per-layer kernel launches from Python, and for small models at small-to-moderate batch sizes the CPU launch overhead becomes the dominant cost of the step, leaving the GPU idle between kernels. vLLM's engine captures the decode computation as CUDA graphs for a set of batch sizes during warmup and replays the matching graph at runtime, collapsing hundreds of launches into one submission. The measured effect is exactly the before/after in figure three: the GPU-idle gaps close and per-token latency drops, with the largest gains on smaller models and faster GPUs where launch overhead is the biggest share. The lesson for profiling is that `enforce_eager=True` is a fantastic *diagnostic* — flip it on, profile, and the launch-bound gaps become dramatically visible, which confirms how much the graphs are buying you before you commit to the (shape-rigid) captured path. The [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) covers the engine internals; the point here is that a documented, shipped optimization exists precisely because this profile signature is so common.

**FlashAttention and the memory wall.** FlashAttention (Dao, Fu, Ermon, Rudra, Ré, 2022) is the canonical example of a profile-driven kernel redesign. Standard attention computes the $S \times S$ score matrix, writes it to HBM, reads it back for the softmax, writes again, and reads again for the value multiply — its cost is dominated by that $O(N^2)$ HBM traffic, so it is memory-bound despite being nominally a compute operation. A profile shows exactly this: low tensor-core utilization, high memory throughput, the compute units starved. FlashAttention tiles the computation and performs an online softmax so the score matrix is never materialized in HBM — the intermediates stay in on-chip SRAM — which slashes the byte traffic and moves the operation toward compute-bound. FlashAttention-2 (Dao, 2023) went further on the *profile* side: it reduced the number of non-matmul FLOPs and improved warp-level work partitioning to cut shared-memory traffic, raising achieved throughput to roughly 50–73% of the A100's theoretical peak FLOP/s — about 2× over the first version — at moderate occupancy. That "high throughput at moderate occupancy" result is the concrete refutation of the "maximize occupancy" instinct, and it is exactly what a per-kernel `ncu` report on FA-2 shows. When you profile an attention kernel and see tensor cores well-used and memory throughput moderate, that is FA-2 working; when you see the opposite, you are on an unfused path and the fix is to adopt a proper fused attention backend, covered in the [attention backends deep dive](/blog/machine-learning/model-serving/attention-backends-deep-dive-flashattention-flashinfer).

**Tensor-parallel comms bubbles.** In multi-GPU tensor-parallel serving, the all-reduce after each attention and MLP block is a synchronization point, and whether it is overlapped with compute is directly visible in an `nsys` multi-stream timeline. NVIDIA's own guidance for profiling TensorRT-LLM and Megatron-style serving uses exactly this view: capture with `--trace=cuda,nvtx,nccl` across ranks, and look for NCCL spans where the compute stream sits idle. When they do not overlap, every rank pays the full collective latency as bubble, and at TP=8 across NVLink that bubble can be a meaningful fraction of the decode step. The fixes — overlapping the collective on a separate stream, or reducing the TP degree so each rank does more compute per collective — are chosen by *measuring the bubble*, not by assuming. The [running vLLM distributed in production](/blog/machine-learning/model-serving/running-vllm-distributed-in-production) post covers the deployment side; the profile is what quantifies whether the comms are actually your bottleneck or a red herring. A common surprise here is that reducing TP (fewer GPUs) can *improve* latency for a model that fits in fewer cards, because it removes the collective entirely — a conclusion you would never reach without seeing the bubble on the timeline.

**The host-side bottleneck nobody profiles.** A recurring real-world finding is that on a fast GPU with an expensive sampler, TPOT is dominated by CPU work the kernel profiler never sees. Structured-output decoding (constraining generation to a JSON schema or regex) runs a mask computation on the CPU each step; a naive implementation can cost more per token than the entire transformer forward pass. A py-spy flamegraph of the running server makes this obvious in thirty seconds — the flame is dominated by the constraint checker, not by anything CUDA — and the fix is algorithmic (precompiled FSM masks, moving the check off the critical path), not a kernel change. This is the case that most vindicates the *ladder*: if you had jumped straight to `ncu`, you would have found perfectly healthy kernels and concluded, wrongly, that there was nothing to fix. The coarse rungs exist precisely to keep you from optimizing a GPU that is not the problem.

**Chunked prefill and prefill-decode interference.** A subtler pattern surfaces only on a multi-stream `nsys` timeline of a server running mixed traffic: a long prefill for one incoming request lands in the same batch iteration as the decode steps of many others, and the prefill's big compute-bound GEMMs stretch that iteration, spiking the TPOT of every decoder that shared it. On the timeline it reads as a periodic *widening* of the decode-step spacing that correlates exactly with prefill GEMMs appearing on the compute stream — the decodes are not intrinsically slower, they are queued behind a fat prefill. The fix is chunked prefill: split a long prefill into bounded token-count chunks so each iteration carries at most a fixed prefill budget alongside the decodes, capping the worst-case step width. The profile is what proves the interference is real and sets how big the chunk budget must be — measure the step-width distribution before and after, and the long tail of wide steps should collapse back into the body. vLLM's and TensorRT-LLM's scheduler knobs for this (`--max-num-batched-tokens`, `--enable-chunked-prefill`) are tuned by reading exactly this timeline: set the budget too low and you starve prefill throughput, too high and the decode tail returns, and only the measured step-width histogram tells you where the knee sits. This is also a caution about CUDA graphs — a chunked-prefill iteration has a variable token count, so it does not fit the fixed-shape graph the pure-decode path replays, which is why serving engines graph the decode steps and run prefill eagerly. That split becomes obvious the moment you put the two phases' kernel-shape distributions side by side on the same trace.

## When to use this (and when not to)

Kernel-level profiling is powerful and it is expensive in engineer-time, so the decision of when to reach for it is itself part of the craft. The honest rule is: **do not micro-profile before the coarse metrics point somewhere.** If your Prometheus dashboard shows the request queue backing up and the KV-cache at 100%, the answer is capacity or scheduling, and no `ncu` report will help — you would spend a day producing a beautiful kernel analysis of a system whose bottleneck is that it is out of memory. Rung one exists to prevent exactly this. Descend only when the coarse metric says the *execution* of a step is the problem and the queue is not.

Do not descend to `ncu` (rung four) until `nsys` (rung three) has named a specific kernel. `ncu --set full` against a whole workload is hours of replay for a report too large to read. The 100–1000× overhead is not a nuisance to tolerate; it is a signal that this tool is for one kernel at a time. If you find yourself running `ncu` without a specific kernel name in mind, you have skipped a rung.

Be honest about **profiling overhead distorting the measurement.** `torch.profiler` with `with_stack=True` and `record_shapes=True` can add 1.5× or more, and that overhead is not uniform — it falls more heavily on the CPU side, which can make a GPU-bound workload *look* more CPU-bound than it is. `nsys` is lighter but `--gpu-metrics-device` sampling still perturbs timing slightly. And `ncu`'s clock-locking means its absolute latencies are base-clock, not the boosted numbers you see in production. Always cross-check a profiler's story against an unperturbed wall-clock measurement (a plain timing loop) before you trust a surprising conclusion. If the profiler says the CPU is the bottleneck, confirm it by checking that reducing CPU work actually moves end-to-end TPOT.

Do not profile at all when the win is not worth the shape rigidity. The classic example is CUDA graphs: they fix launch-bound decode beautifully, but they require fixed shapes and no data-dependent control flow, which complicates features like variable batch sizes, LoRA hot-swapping, and speculative decoding. If your decode step is only 15% launch-bound and you rely heavily on dynamic behavior, the profile says "launch-bound" but the *right decision* might still be to leave it eager. The profile identifies the bound; the SLO triangle and your feature requirements decide whether the fix is worth its cost. This is also why the [autotuning serving configs for your workload](/blog/machine-learning/model-serving/autotuning-serving-configs-for-your-workload) post treats the profile as an input to a search over configurations, not as a mandate — the same measurement can justify different choices depending on what you are optimizing for.

Finally, know when the tool is simply wrong for the layer. Kernel profiling answers "why is this GPU step slow." It does not answer "why is my p99 across the fleet bad" (that is load balancing, autoscaling, and tail-at-scale — see [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management)), and it does not answer "why does this request return the wrong tokens" (that is a correctness bug, not a performance one). Match the tool to the question.

## Key takeaways

- **Never guess the bottleneck — measure it.** A serving workload is launch-, memory-, or compute-bound, and a profile tells you which in minutes. Guessing wrong costs sprints.
- **Climb the ladder in order.** Coarse Prometheus metrics → `torch.profiler` → `nsys` timeline → `ncu` single-kernel. Each rung is more expensive and narrower; descend only when the rung above points somewhere specific.
- **Profile steady state, not cold start.** Warm up under representative load, skip the first hundreds of iterations, and bracket a bounded window (`schedule`, `--capture-range`, `--launch-skip`). Cold-start and warmup numbers are lies.
- **Read the timeline for gaps, read the kernel for counters.** `nsys` shows time *between* kernels — launch gaps, comms bubbles, CPU-GPU overlap. `ncu` shows what is happening *inside* one kernel — DRAM%, occupancy, tensor-core%.
- **DRAM throughput % and SM occupancy are the two you must read correctly.** High DRAM% near peak = memory-bound near floor (fix by moving fewer bytes: fuse/quantize). High occupancy does *not* mean fast; occupancy only matters when it is low enough (<30%) to fail to hide latency.
- **The same symptom has three fixes.** A slow kernel is fixed differently if it is compute-, memory-, or occupancy-bound. Only the profile distinguishes them, so the profile — not intuition — chooses the fix.
- **Launch-bound decode is the most common LLM-serving profile signature, and it worsens on faster GPUs.** CUDA graphs are the fix, and `enforce_eager=True` is the diagnostic that reveals the win.
- **Do not forget the host side.** py-spy on the running process catches CPU-bound sampling, detokenization, and constrained-decoding costs that every GPU counter misses.
- **Fix one bound, then re-measure with the identical capture.** The re-measure catches the case where the fix shifted the bottleneck instead of removing it.

## Further reading

- **NVIDIA Nsight Systems User Guide** — the authoritative reference for `nsys` flags, the timeline UI, `--gpu-metrics-device`, capture ranges, and `nsys stats` reports. (docs.nvidia.com/nsight-systems)
- **NVIDIA Nsight Compute Documentation** — the metric reference (the `sm__` and `gpu__dram__` identifiers), the Speed-of-Light and roofline sections, kernel replay, and clock control. (docs.nvidia.com/nsight-compute)
- **PyTorch Profiler and HTA (Holistic Trace Analysis) documentation** — `torch.profiler` API, `schedule`, `record_function`, TensorBoard/Perfetto trace export, and automated trace analysis. (pytorch.org/docs/stable/profiler.html)
- **Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)** — the IO-aware, memory-bound analysis of attention that this whole roofline framing rests on.
- **Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)** — the work-partitioning and occupancy result that refutes the "maximize occupancy" instinct.
- **Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009)** — the original roofline paper; the ridge-point mechanics used throughout.
- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)** — the vLLM paper; context for why decode is memory-bound and where CUDA graphs help.
- Within this series: [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference), [custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference), [kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile), [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving), and [tracing and debugging LLM serving](/blog/machine-learning/model-serving/tracing-and-debugging-llm-serving).
