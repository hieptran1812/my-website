---
title: "CUDA graphs in PyTorch: torch.cuda.graph, make_graphed_callables, and static buffers"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "The runnable side of CUDA graphs: how to capture a forward pass with torch.cuda.graph, the static-buffer discipline that makes replay correct, when to reach for make_graphed_callables or reduce-overhead instead, and how to prove in a trace that the launches collapsed."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "cuda-graphs",
    "pytorch",
    "cuda",
    "profiling",
    "latency",
    "throughput",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 41
---

You already know *why* CUDA graphs work. You have read the [first-principles post](/blog/machine-learning/performance-engineering/cuda-graphs-from-first-principles): a graph is a recorded DAG of kernels that the driver can replay with a single host call, and that single call is how you stop paying five-to-ten microseconds of CPU launch overhead for every one of the thousands of tiny kernels a model fires per step. You have seen the [launch-overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) that makes an A100 sit at 34% utilization while Python scrambles to enqueue work fast enough to keep it fed. The theory is settled. This post is about making PyTorch actually do it — the API, the discipline, and the exact code you paste into a real forward pass.

There are two ways in, and they sit at opposite ends of a control-versus-convenience spectrum. The low-level one is the `torch.cuda.graph(g)` context manager: you allocate the buffers, run the warmup, drive the capture, and manage the replay loop by hand. The high-level one is `torch.cuda.make_graphed_callables`, which wraps a module so it graphs itself transparently, warmup and capture included, and is the right tool for a training step. And there is a one-line shortcut, `torch.compile(mode="reduce-overhead")`, that fuses *and* graphs for you. By the end you will be able to write all three, know which to pick, and — the part that separates a working graph from a hang or a garbage tensor — hold the **static-buffer discipline** in your hands: the small, unforgiving rule that you must copy new data *into* the same tensors the graph recorded and read results *out of* the same tensors, every single step, forever.

![a horizontal lifecycle showing static buffers allocated then a warmup phase on a side stream then a one time capture then a repeating copy and replay step](/imgs/blogs/cuda-graphs-in-pytorch-1.webp)

The figure above is the whole lifecycle on one line, and it is worth fixing in your head before any code, because it explains where every rule comes from. Notice the asymmetry: the first three stages — allocate the static input and output once, warm up for a handful of iterations on a side stream so cuBLAS and cuDNN finish their autotuning and the allocator stops growing, then capture the kernel DAG a single time — all happen exactly *once*, at setup. Only the last stage repeats. And that repeating stage is trivially cheap: copy the new batch into the fixed input buffer, call `g.replay()`, done. All the expense has been front-loaded out of the hot path. That shape — pay once, replay forever — is why a graphed service can drop its host cost by 40x. Everything in this post is either one of those four stages or a failure mode of getting one of them wrong. This is the third track of the [performance playbook](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the fixes — and CUDA graphs are the first heavy one.

## Two APIs and a shortcut: which one you actually want

Before we open the manual API, it helps to see the three tools side by side, because reaching for the wrong one wastes an afternoon. The choice is not about which is "best" — it is about how much control you need versus how much ceremony you are willing to write yourself.

![a comparison grid of three pytorch cuda graph apis rated on how much manual control they give whether they handle warmup and what each is best suited for](/imgs/blogs/cuda-graphs-in-pytorch-2.webp)

The matrix lays out the trade. `torch.cuda.graph(g)` is the raw context manager: maximum control, and *you* own every step — you allocate the static tensors, you run the warmup, you decide what gets captured, you write the replay loop. It is the right tool when your forward pass has structure the automated wrappers can't guess: a custom sampling loop, a KV-cache update, a hand-rolled decode step, anything where you need to reach inside the captured region. `make_graphed_callables(module, sample_args)` is the batteries-included wrapper: you hand it a module and a representative input, it runs warmup and capture internally, returns a drop-in callable, and — crucially — it captures the **backward pass too**, which makes it the natural choice for a training step where you'd otherwise have to capture forward and backward separately by hand. And `torch.compile(model, mode="reduce-overhead")` is the shortcut that does compilation *and* CUDA graphs in one call: Inductor fuses the elementwise chains into fewer, bigger kernels, then the runtime wraps the result in graphs. Least control, least code, and it composes both fixes at once — but it inherits `torch.compile`'s recompilation behavior on shape changes, which we'll return to.

Here is the same decision as prose, because it's the one thing to get right:

| If you need… | Reach for | Why |
|---|---|---|
| A custom loop, KV cache, or full control of the captured region | `torch.cuda.graph(g)` | You drive warmup, capture, and the static-buffer loop yourself |
| A training step graphed without hand-writing backward capture | `make_graphed_callables` | It captures forward and backward and handles warmup for you |
| The biggest single-line win, fusion plus graphs, for a fixed-shape service | `torch.compile(mode="reduce-overhead")` | Inductor fusion then automatic graphing; least code |
| A model whose shape changes every request | *None of the above, yet* | Graphs are shape-locked; bucket shapes first (see the serving section) |

We'll build the manual API first, in full, because it is the one that teaches you *why* the static-buffer discipline exists. Once you have felt that by hand, the automated wrappers stop being magic — they are just doing the same four stages you did, behind a function call.

## The manual API: torch.cuda.graph, step by step

Here is a complete, runnable manual capture around an inference forward pass. Read it top to bottom once; then we'll take each stage apart and derive why it has to be exactly this way.

```python
import torch

model = build_model().cuda().eval()          # your model, in eval mode

# ---- Stage 1: allocate STATIC I/O once. These pointers never change. ----
# Fixed shape: batch 8, sequence length 128, integer token ids.
static_in = torch.zeros(8, 128, dtype=torch.long, device="cuda")
static_out = None                             # bound by capture, below

# ---- Stage 2: warm up on a SIDE stream so autotune + allocator settle. ----
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())    # side stream waits for pending work
with torch.cuda.stream(s):
    for _ in range(5):                        # 3 to 11 iters is plenty
        with torch.no_grad():
            static_out = model(static_in)
torch.cuda.current_stream().wait_stream(s)    # default stream waits for warmup

# ---- Stage 3: capture the kernel DAG a single time. ----
g = torch.cuda.CUDAGraph()
with torch.no_grad():
    with torch.cuda.graph(g):
        static_out = model(static_in)         # records launches; runs no real step for us

# ---- Stage 4: per request, copy in -> replay -> read out. Same tensors. ----
@torch.no_grad()
def infer(new_tokens):                         # new_tokens: shape (8, 128), long, cuda
    static_in.copy_(new_tokens)                # write INTO the buffer the graph recorded
    g.replay()                                 # one host launch; replays all kernels
    return static_out.clone()                  # read OUT of the recorded output buffer
```

Nine lines of setup and a three-line hot loop. That is the entire manual API. Every line is load-bearing, so let's go stage by stage — and where a stage exists to prevent a specific disaster, I'll name the disaster.

### Stage 1: static buffers, and why they are non-negotiable

The single most important idea in this whole post lives here, so it deserves a careful statement. When the driver captures a graph, it records the *exact device memory addresses* of every tensor each kernel reads and writes. A captured kernel is not "run softmax on whatever tensor is called `scores`" — it is "run softmax on the 288 KB starting at device address `0x7f2a...`". Replay re-issues those recorded kernels against those recorded addresses, verbatim, with zero re-planning. That is precisely why replay is one cheap host call instead of thousands: there is nothing left to decide, only pointers to fire.

The consequence is strict. If, on step two, your input tensor is a *fresh* allocation at a *different* address — which is exactly what `x = new_batch.cuda()` gives you, a brand-new tensor every call — the graph will happily replay against the *old* address from capture time, read whatever stale bytes happen to live there now, and produce a confident, wrong answer. No exception, no warning. Just garbage. The defense is the static buffer: allocate `static_in` once, and on every step `copy_` the new data *into* it rather than binding a new tensor. The pointer the graph recorded stays valid because the tensor behind it never moves.

The output side is symmetric. During capture, `static_out = model(static_in)` rebinds the Python name `static_out` to the tensor the model's final kernel wrote into — a tensor that lives inside the graph's memory region. After capture, `g.replay()` writes fresh results into that *same* tensor every time. So you read results by looking at `static_out`, and if you need to keep them past the next replay, you `.clone()` them out first, because the next `replay()` will overwrite that buffer in place. Forget the clone and step N+1 stomps step N's output before you've used it.

![a layered view of the four things that must keep fixed addresses the static input and output tensors the graph object and its private memory pool sitting on the caching allocator](/imgs/blogs/cuda-graphs-in-pytorch-4.webp)

The stack above is the full inventory of what has to hold still, and it answers the question people ask right after "why static buffers?" — namely, *what else?* Four things must keep fixed addresses across replays. The **static input and output tensors**, for the reason we just derived. The **model weights**, which is usually free — weights are allocated at model construction and don't move — but bites you if you re-materialize weights, swap in LoRA adapters, or move the model to a different device after capture. The **graph object** `g` itself, which owns the recorded DAG. And the **private memory pool** the graph allocates its internal activations from, which sits on top of the normal caching allocator and which we'll unpack in its own section. The one-line takeaway from the figure: a graph is a photograph of pointers, and everything in the photograph must still be there when you replay it.

### Stage 2: warmup on a side stream, and why capture would otherwise hang

Warmup is the stage people delete to "save startup time," and then they file a bug that says "torch.cuda.graph hangs" or "my captured graph errors on the first replay." The warmup is not optional ceremony; it exists to move three categories of one-time, non-capturable work *out* of the capture region.

First, **library autotuning**. The first time cuBLAS or cuDNN sees a new matmul or convolution shape, it runs heuristics — sometimes it literally benchmarks a few candidate kernels — to pick the fastest implementation. That selection process launches its own kernels, allocates scratch, and generally does things that are illegal to record inside a graph capture. Run it during warmup, for the exact shapes you'll capture, and by capture time the library has cached its choice and simply issues the chosen kernel. Second, **allocator growth**. A cold caching allocator has no free blocks, so the first forward triggers a cascade of `cudaMalloc` calls to grow the pool — and `cudaMalloc` is one of the operations you cannot perform inside a capture. Warmup grows the pool to steady state so capture finds every block it needs already resident. Third, **lazy initialization** — cuDNN handles, cuBLAS workspaces, autograd machinery, anything a module sets up on first call. Warmup pays it all up front.

Why a *side* stream? Graph capture works by putting a specific CUDA stream into capture mode; every kernel launched on that stream while it's capturing gets recorded instead of executed. PyTorch's default stream (and the legacy default stream) carry a lot of implicit, library-level traffic, and capturing on it risks recording — or colliding with — work you didn't intend. Capturing on a fresh side stream isolates the recording to exactly your forward pass. The two `wait_stream` calls are the handshake that keeps it correct: `s.wait_stream(current_stream())` makes the side stream wait for any already-queued work to finish before warmup begins, and `current_stream().wait_stream(s)` makes the default stream wait for warmup to complete before you proceed to capture. Skip those and you get a race between warmup and capture that manifests as exactly the intermittent hang people report. (The `torch.cuda.graph` context manager runs the capture itself on an internal side stream, so you don't manage the capture stream by hand — but you *do* manage the warmup stream, which is what the code above shows.)

How many warmup iterations? Three to eleven. The lower bound covers plain inference — a couple of passes to trigger autotuning and grow the allocator. The upper bound, eleven, is what PyTorch's own `make_graphed_callables` uses by default, because training has more to settle: the autograd graph gets built, gradient buffers get allocated, and optimizer state materializes. When in doubt, more warmup only costs startup time, never correctness; too little risks capturing an un-settled state.

### Stage 3: capture, and what "recording" actually does

The capture itself is anticlimactic, which is the point: `with torch.cuda.graph(g): static_out = model(static_in)`. Inside that context, the forward pass runs, but instead of *executing* on the GPU it is *recorded* — every `cudaLaunchKernel` that PyTorch issues is intercepted by the driver and appended to the graph `g` as a node, with its arguments (including those all-important pointers) frozen. When the context exits, `g` holds the complete kernel DAG of one forward pass. Nothing has run for you to consume; you've only built the recording.

![a dataflow where a new request tensor is copied into a fixed static input which together with the recorded graph feeds a single replay call that writes a fixed static output read back as the response](/imgs/blogs/cuda-graphs-in-pytorch-3.webp)

The figure captures the steady-state loop that stage 4 runs, and it makes the branch structure explicit in a way the linear code hides. A new request tensor — dynamically allocated, different every call — is *copied into* `static_in`, the fixed buffer. That fixed input and the recorded graph object `g` both feed into the single `g.replay()` call: the replay reads from the pointer it recorded (which is `static_in`), and it fires the DAG that `g` holds. Replay writes into `static_out`, also a fixed pointer, and you read the response out of there. The two arrows merging into `replay` are the heart of it: replay needs *both* the fresh data sitting in the recorded input buffer *and* the recorded graph, and it touches nothing else. Everything dynamic about a request has to funnel through that `copy_` into the static buffer, because the static buffer is the only place the graph knows how to look.

### The mechanism: why this is fast, and how fast it can possibly be

Let's make the win quantitative, because it tells you both *why* graphing helps and — more usefully — *when it can't*. Recall the launch-overhead accounting from the [launch-overhead post](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem). A step that launches $N$ kernels, each costing $t_\text{launch}$ of host time to enqueue, spends

$$t_\text{host} = N \cdot t_\text{launch}$$

on the CPU just issuing launches. With $N = 1800$ kernels and $t_\text{launch} \approx 6.7$ µs, that's about 12 ms of pure host enqueue work per step. Replay replaces all $N$ launches with a *single* host call — one `cudaGraphLaunch` — whose cost is essentially one launch's worth, a few microseconds plus driver overhead, call it 0.3 ms including the copy_. The host cost goes from $N \cdot t_\text{launch}$ to roughly $t_\text{launch}$: a factor of $N$ on the host side, 40x here.

But the step time is not the host time — it's whichever of host and device is the bottleneck, because they overlap. Before graphing, if the host needs 12 ms to feed a device that only has 6.2 ms of work, the device idles and the step is host-bound at roughly 12 ms (plus the gap-induced serialization that pushes p50 higher). After graphing, the host needs 0.3 ms, so the step is *device*-bound at 6.2 ms. The device work itself does not change — graphing records the same kernels — which is why the honest speedup is bounded by how host-bound you started.

This is Amdahl's law wearing a CUDA hat, and it's the single most important thing to internalize before you graph anything. If the launch overhead is a fraction $p$ of your step time, then driving that overhead to zero gives a *maximum* speedup of

$$S_\text{max} = \frac{1}{1 - p}.$$

If launch overhead is 60% of your step ($p = 0.6$), the ceiling is 2.5x — and you'll get close, because graphs really do drive $p$ toward zero. But if you are already device-bound, with launch overhead at 10% of the step ($p = 0.1$), the ceiling is 1.11x, and after you subtract capture complexity and the static-buffer bookkeeping, graphing may not be worth it at all. **Profile first.** The [profiler footer](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) — Self CPU time total versus Self CUDA time total — is exactly the measurement of $p$: when CPU dwarfs CUDA, $p$ is large and graphs will pay; when they're close, $p$ is small and you should look elsewhere.

## Measuring the win: before and after on an A100

Theory predicts a large host-side win for a launch-bound service. Let's measure it, honestly, on named hardware, using the same clean-benchmark hygiene the [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) insists on: warm up, lock clocks, `torch.cuda.synchronize()` before reading the clock, time steady-state steps with CUDA events, and measure the before/after *outside* the profiler so the profiler's own overhead doesn't leak into the number.

![two panels contrasting an eager forward firing eighteen hundred host launches at twelve milliseconds against a graphed forward that fires a single replay at a third of a millisecond](/imgs/blogs/cuda-graphs-in-pytorch-5.webp)

The two panels are the shape of the result before we look at the exact numbers, and they encode the one thing you must not misread: *the device work is the same on both sides.* Graphing did not make the GPU faster. It collapsed 1800 host launches into 1 replay, cut host time from 12.0 ms to 0.3 ms, and by keeping the launch queue full it let utilization climb from 34% to 85%. The GPU was always capable of the work; it was starved, and the graph fed it. Everything won here came from the host side of the ledger, which is why the fix is free of any accuracy cost — the math is byte-for-byte identical, only the scheduling changed.

#### Worked example: graphing a launch-bound transformer classifier

**Symptom.** A six-layer transformer classifier serves on a single A100 80GB SXM (312 dense bf16 TFLOP/s, 2.0 TB/s HBM2e). At batch 8, sequence length 128, `nvidia-smi` shows GPU utilization hovering near 34%, p50 latency 15.8 ms, throughput 505 req/s. On-demand A100 runs about \$2.20 per GPU-hour, so 505 req/s is roughly \$1.21 per million requests. The standing suggestion is "buy an H100."

**Profile.** Wrap the forward in `torch.profiler` with `with_stack=False`, sort by `cpu_time_total`. The footer reads `Self CPU time total: 120.4ms` across 10 steps and `Self CUDA time total: 62.1ms`, and the `cudaLaunchKernel` row shows 18,000 calls — 1800 per step. CPU is roughly 2x CUDA and the launch row is the largest single consumer of host time. This is the launch-bound signature, unambiguously: $p \approx 0.5$, so Amdahl's ceiling is about 2x on latency, with a large utilization gain on top.

```console
--------------------------  ------------  ------------  ------------  ------------
                      Name    Self CPU %      Self CPU     Self CUDA    # of Calls
--------------------------  ------------  ------------  ------------  ------------
          cudaLaunchKernel        41.2%      11.80ms       0.000us         18000
               aten::addmm         6.1%       1.75ms       13.54ms          1920
            aten::_softmax         2.9%       0.83ms       11.95ms           360
   aten::native_layer_norm         5.4%       1.55ms        6.40ms           720
                 aten::gelu         2.2%       0.63ms        5.41ms           360
--------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 120.4ms
Self CUDA time total: 62.1ms
```

**Hypothesis.** The device does 6.2 ms of real work per step, but the step takes ~15 ms because the host needs ~12 ms to enqueue 1800 tiny kernels and the GPU idles in the gaps. An H100 would run the same 6.2 ms of work slightly faster and then idle *even more*, at twice the price — strictly worse dollars-per-request. This is a launch-count problem, and the fix is a CUDA graph.

**Fix.** The manual capture from the section above: static `(8, 128)` input, five warmup iters on a side stream, one capture, then the `copy_` / `replay` / `clone` loop. No architecture change, no new hardware.

**Re-measure**, same A100, honest out-of-profiler timing:

| Metric | Before (eager) | After (manual graph) |
|---|---|---|
| GPU utilization | 34% | 85% |
| Host time / step | 12.0 ms | 0.3 ms |
| Device time / step | 6.2 ms | 6.2 ms |
| Host launches / step | 1800 | 1 (graph replay) |
| p50 latency | 15.8 ms | 6.9 ms |
| p99 latency | 34.0 ms | 8.3 ms |
| Throughput @ batch 8 | 505 req/s | 1150 req/s |
| Cost per million req | \$1.21 | \$0.53 |

The row that tells the truth is *device time / step*: 6.2 ms before and after, unchanged. The GPU was never the bottleneck. Host time fell from 12.0 ms to 0.3 ms, launches from 1800 to 1, and with the queue no longer starving, utilization more than doubled, p50 halved, p99 collapsed from 34 ms to 8.3 ms — the tail shrank the most, because the eager tail was dominated by unlucky steps where the host fell furthest behind — and throughput went 2.3x, on the *same card*, at 44% of the cost per request. The H100 was a \$1.21 answer to a \$0.53 software fix.

**How we timed it honestly.** Latency numbers came from CUDA events, not `time.time()`, so we measure device-inclusive wall time without a blocking sync in the loop:

```python
import torch, numpy as np

def bench(fn, batch, iters=200, warmup=50):
    for _ in range(warmup):                    # steady state before timing
        fn(batch)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(batch)
        ends[i].record()
    torch.cuda.synchronize()                   # one sync AFTER the loop, not inside
    ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return np.percentile(ms, 50), np.percentile(ms, 99)

p50_eager, p99_eager = bench(lambda b: model(b), batch)
p50_graph, p99_graph = bench(infer, batch)     # infer() is the replay loop above
```

The single sync sits *after* the timing loop, never inside it — a per-iteration sync would serialize host and device and destroy the overlap you're trying to measure, inflating the eager number and understating the win. That is the discipline: measure the thing, not the measurement.

## Graphing a training step by hand (and why you probably shouldn't)

Everything so far graphed an inference forward. Training is where launch overhead often hurts *most* — a plain Adam step launches a handful of tiny elementwise kernels per parameter tensor, so a model with hundreds of parameters fires hundreds of microscopic kernels every step on top of the forward and backward — but it's also where the manual API gets genuinely fiddly, and seeing that fiddliness is the best argument for the wrapper we're about to reach. To graph training by hand you capture the forward *and* the backward together, because the backward's kernels are just as launch-bound as the forward's:

```python
static_x = torch.zeros(32, 3, 224, 224, device="cuda")
static_y = torch.zeros(32, dtype=torch.long, device="cuda")

# Warmup must include BACKWARD so autograd + gradient buffers settle.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(11):                          # training settles slower: 11 iters
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(static_x), static_y)
        loss.backward()
torch.cuda.current_stream().wait_stream(s)

# Capture forward + backward as one region. Gradients fill the .grad buffers.
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)            # allocate .grad buffers ONCE, up front
with torch.cuda.graph(g):
    static_loss = criterion(model(static_x), static_y)
    static_loss.backward()                       # records backward launches too

for x, y in loader:                              # x: (32,3,224,224), y: (32,)
    static_x.copy_(x); static_y.copy_(y)         # copy real data INTO static buffers
    g.replay()                                   # replays fwd + bwd; fills .grad in place
    optimizer.step()                             # step runs OUTSIDE the graph
    for p in model.parameters():                 # re-zero WITHOUT reallocating .grad
        if p.grad is not None:
            p.grad.zero_()                        # keep the same grad storage the graph wrote
```

The last two lines are the whole trap. During capture, backward wrote gradients into a specific set of `.grad` tensors at specific addresses; replay writes into those *same* addresses every step. So between steps you must zero the gradients *in place* — `p.grad.zero_()` — and never `zero_grad(set_to_none=True)`, because setting them to `None` frees the buffers and the next `optimizer.step()` (and the next replay) would find different or missing storage. That single asymmetry — `set_to_none=True` is the recommended default *everywhere else* and is exactly wrong here — is representative of the whole manual-training experience: correct, but studded with sharp edges where a habit from eager code silently breaks the graph. Every one of those edges is a place `make_graphed_callables` gets right for you, which is why, for training, you almost always want the wrapper instead.

## The high-level API: make_graphed_callables

The manual API earns its keep when you need control, but for the common case — "graph this module, warmup and all" — writing nine lines of stream handshaking by hand is error-prone ceremony. `torch.cuda.make_graphed_callables` does the whole lifecycle for you and hands back a drop-in callable.

```python
from torch.cuda import make_graphed_callables

model = build_model().cuda().eval()

# One representative sample per positional arg. Shape + dtype must match
# what you will actually feed at runtime.
sample = torch.zeros(8, 128, dtype=torch.long, device="cuda")

graphed_model = make_graphed_callables(model, (sample,))

# Use it exactly like the original module. Warmup and capture already happened.
out = graphed_model(real_batch)                # real_batch must be shape (8, 128)
```

That's it. Internally it does the same four stages you did by hand: it allocates static placeholders shaped like your `sample`, runs warmup iterations on a side stream (eleven by default), captures, and returns a callable whose `__call__` copies the argument into the static buffer, replays, and returns the static output. You never see the stream handshake or the `copy_`; it's all inside.

The reason `make_graphed_callables` is the *training* tool, not just a convenience, is that it captures the **backward pass** as well. Pass a module whose sample args have `requires_grad=True` and it captures two graphs — one for forward, one for backward — and wires them so that a normal `loss.backward()` replays the backward graph. Doing that by hand with `torch.cuda.graph` is genuinely fiddly: you'd have to capture forward and backward as separate graphs, manage the gradient static buffers, and get the autograd interaction exactly right. `make_graphed_callables` is that fiddliness packaged correctly. A graphed training step looks like this:

```python
# sample must require grad so backward is captured too.
sample = torch.randn(32, 3, 224, 224, device="cuda", requires_grad=True)
model = make_graphed_callables(model, (sample,))

for x, y in loader:                            # x must be shape (32, 3, 224, 224)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)              # forward graph replays
    loss.backward()                            # backward graph replays
    optimizer.step()
```

You can also graph *parts* of a model rather than the whole thing — pass a tuple of modules and a tuple of sample-arg tuples, and it graphs each independently. That matters when only some submodules have static shapes: graph the fixed-shape transformer blocks, leave a variable-length embedding lookup eager. The trade you're making versus the manual API is control for safety: you can't reach inside the captured region to, say, update a KV cache between sub-steps, and the shape is pinned to whatever `sample` you passed. For a straightforward "make my training step stop being launch-bound," it's the right default and it's far harder to get wrong than the manual loop.

| | `torch.cuda.graph` | `make_graphed_callables` | `reduce-overhead` |
|---|---|---|---|
| Warmup | you write it | automatic (11 iters) | automatic |
| Capture | you drive it | automatic | automatic |
| Backward pass | manual, separate graph | captured for you | captured for you |
| Static buffers | you allocate + `copy_` | hidden inside | hidden inside |
| Fusion | none (records as-is) | none | yes (Inductor) |
| Reach inside region | full | none | none |
| Fails loudly on shape change | you handle it | re-capture needed | recompiles |
| Best for | custom loops, KV cache | training steps | fixed-shape inference |

## The graph memory pool, and why you can't malloc inside

There's a piece of machinery under both APIs worth understanding, because it's the source of the "you can't free or reallocate inside a graph" rule and it foreshadows the whole [memory track](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator) of this series. A captured graph doesn't just record kernel launches — it records the memory those kernels touch, including the intermediate activations produced *during* the forward pass. Those activations have to live somewhere with stable addresses across replays, so PyTorch gives each graph a **private memory pool**: a region the caching allocator sets aside, from which the graph's internal tensors are served, and which is *not* recycled back into the general allocator between replays.

The reason `cudaMalloc` and `cudaFree` are illegal inside a capture follows directly. Capture records operations to *replay*; a malloc during capture would have to be replayed too, but replay must hit the exact same addresses every time, and a real allocator can't promise that — it might hand back a different block on replay, or the block it recorded might now belong to something else. So the CUDA graph API forbids allocation and free inside the captured region, and PyTorch satisfies that by pre-growing the pool during warmup (that's the second job warmup does) so that capture finds every block it needs already carved out. This is why a cold, un-warmed capture hangs or errors: it tries to grow the allocator mid-capture, which is exactly the forbidden operation.

You can share one pool across multiple graphs when you know they never replay concurrently, which saves memory — two graphs that each need 2 GB of activation scratch can share a single 2 GB pool if they run one-at-a-time. The handle is `torch.cuda.graph_pool_handle()`:

```python
pool = torch.cuda.graph_pool_handle()          # a shareable private pool

g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1, pool=pool):
    out1 = model(static_in_1)

g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2, pool=pool):          # reuses the SAME pool
    out2 = model(static_in_2)
```

The saving is real for a serving loop that captures one graph per batch bucket (coming up next): if the buckets never run at the same time, they can share a pool instead of each reserving its own peak footprint. The cost of getting it wrong is a correctness bug — two graphs sharing a pool *and* replaying concurrently would corrupt each other's scratch — so share only when serialized.

The memory cost is not free, and you should measure it, because it's the main thing a graph *takes* in exchange for the launch-overhead it gives back. A graph holds its activation pool for its entire lifetime — it never releases scratch between replays the way eager execution does — so a service that captures four bucket graphs reserves four peak footprints (or one shared one) that stay resident. Read it directly off the allocator before and after capture:

```python
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_reserved() / 1e9
g = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    static_out = model(static_in)
after = torch.cuda.memory_reserved() / 1e9
print(f"graph pool reserved ~{after - before:.2f} GB")   # e.g. 1.8 GB for this bucket
```

`memory_reserved()` is what the allocator holds from the driver, `memory_allocated()` is what's live inside it; the gap between them is the graph's private pool plus any fragmentation. The [caching-allocator post](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator) goes deep on `PYTORCH_CUDA_ALLOC_CONF`, `expandable_segments`, and how the graph pool interacts with fragmentation; for now, the rule is: graphs reserve their memory up front, hold it for their lifetime, and cannot allocate or free while replaying. On a memory-tight box, that reserved footprint is the constraint that decides how many bucket graphs you can afford.

## Wiring it into a serving loop: bucket by shape, then combine

A single graph handles a single shape. A real inference service, though, sees a spread of batch sizes — one request here, a dozen there, a burst of sixty under load. You can't capture "the graph"; you capture *a* graph per shape you intend to serve, and route each request to the graph whose shape it matches. This is bucketing, and it's how CUDA graphs survive contact with a real request stream — the full treatment is the [serving-loop post](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop); here's the core pattern.

![a grid pairing four batch size buckets each with its own captured graph and the replay cost that grows from a fraction of a millisecond at small batch to several milliseconds at large batch](/imgs/blogs/cuda-graphs-in-pytorch-7.webp)

The grid shows the structure and the payoff at once. Each column is a batch bucket — 1, 16, 64 — and each gets its *own* captured graph, because each is a different shape and a graph is locked to the shape it recorded. The bottom row is the honest catch: the replay cost grows with the batch. At batch 1 the device work is tiny, so eliminating launch overhead is nearly the whole step and the win is enormous. At batch 64 the device has real work to do — 3.8 ms of it — and launch overhead was already a small fraction, so the graph still helps but the marginal win is modest. Which is the Amdahl point made concrete: the launch-overhead fraction $p$ shrinks as batch grows, so the biggest CUDA-graph wins are at *small* batch, exactly where a latency-sensitive, low-concurrency service lives.

Here's the bucketing pattern in code. Capture a graph per bucket, pad each incoming request up to the nearest bucket, and replay the matching graph.

```python
BUCKETS = [1, 4, 16, 64]                        # batch sizes we support
SEQ = 128

graphs, static_ins, static_outs = {}, {}, {}
for bs in BUCKETS:
    static_in = torch.zeros(bs, SEQ, dtype=torch.long, device="cuda")
    # warmup on a side stream (omitted for brevity — same handshake as before)
    warmup(model, static_in, iters=5)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = model(static_in)
    graphs[bs], static_ins[bs], static_outs[bs] = g, static_in, static_out

def serve(batch):                               # batch: (n, SEQ), n <= max bucket
    n = batch.shape[0]
    bs = next(b for b in BUCKETS if b >= n)      # smallest bucket that fits
    static_ins[bs][:n].copy_(batch)              # fill the real rows
    static_ins[bs][n:].zero_()                   # pad the rest (masked out later)
    graphs[bs].replay()
    return static_outs[bs][:n].clone()           # return only the real rows
```

Two design points. First, you pad *up* to the nearest bucket and mask the padding out downstream, trading a little wasted compute for the ability to reuse a fixed graph — four buckets cover every batch from 1 to 64 with at most 4x padding waste on an odd size, and usually far less. Second, you combine graphs with **autocast** by capturing *inside* the autocast context, so the recorded kernels are the mixed-precision ones:

```python
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    warmup(model, static_in, iters=5)            # autotune the bf16 kernels
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    with torch.cuda.graph(g):
        static_out = model(static_in)            # records bf16 kernels
```

Warm up *inside* autocast too, so cuBLAS autotunes the bf16 kernels you're about to capture — warm up in fp32 and capture in bf16 and you've defeated the warmup.

Bucketing composes with continuous batching, but with a wrinkle worth flagging before you build it. An LLM decode loop is the ideal graph target — the per-token forward is a fixed-shape, launch-bound step you replay thousands of times — but the *sequence length* grows every token and the KV cache grows with it, so the naive "one shape" assumption breaks unless you pad the KV cache to a fixed maximum and mask, exactly as you padded the batch here. That's why production LLM servers capture graphs over a padded, fixed-capacity KV cache and a bucketed batch dimension, replaying the same decode graph across a whole generation while the scheduler swaps requests in and out of the fixed slots. The full pattern — how graphs, continuous batching, and a paged KV cache coexist — is the subject of the [serving-loop post](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) and the [model-serving treatment](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile); the takeaway here is that the same two moves, pad-to-a-bucket and mask, are what let a fixed graph serve a variable world.

#### Worked example: bucketed serving under mixed load on an L4

**Symptom.** The same classifier, now on an L4 (24 GB, ~242 fp16 TFLOP/s, 300 GB/s) behind an autoscaled endpoint. Traffic is bursty: mostly single requests, occasional bursts to batch 32. A single batch-1 graph gives great latency but wastes the GPU on bursts; dynamic batching without graphs is back to launch-bound. p50 at batch 1 is 9.1 ms eager, p99 under burst spikes to 60 ms as the host falls behind.

**Profile.** At batch 1, the footer shows host time roughly 3x device time — deeply launch-bound, because the L4's smaller device does the tiny batch-1 work in ~2.8 ms while the host still needs ~8 ms to launch 1800 kernels. Under a batch-32 burst, the ratio narrows to near 1:1 — the device finally has enough work per launch to hide the overhead.

**Fix.** Capture four bucket graphs (1, 4, 16, 64), route each request to the smallest fitting bucket, pad and mask. Batch-1 requests replay `g1`; a burst of 32 pads to `g64`. Share one memory pool across the buckets since only one replays at a time.

**Re-measure**, on the L4:

| Metric | Batch 1 eager | Batch 1 graphed | Batch 32 burst eager | Batch 32 burst graphed |
|---|---|---|---|---|
| Host time / step | 8.0 ms | 0.3 ms | 9.5 ms | 0.4 ms |
| Device time / step | 2.8 ms | 2.8 ms | 11.0 ms | 11.0 ms |
| p50 latency | 9.1 ms | 3.1 ms | 12.4 ms | 11.3 ms |
| p99 latency | 21 ms | 3.6 ms | 60 ms | 12.1 ms |
| Utilization (steady) | 28% | 78% | 71% | 88% |

Read the two regimes against each other and the Amdahl story is right there in the numbers. At **batch 1**, host was 8.0 ms against 2.8 ms of device work — the step was 74% launch overhead, $p = 0.74$, so the ceiling was ~3.8x and we got p50 from 9.1 to 3.1 ms, close to it. At **batch 32**, host was 9.5 ms against 11.0 ms of device work — the device was *already* the bottleneck, $p$ small, so p50 barely moved (12.4 to 11.3 ms). But look at p99 in the burst column: 60 ms to 12.1 ms. Even when the median is device-bound, the *tail* was still host-driven — the worst steps were the ones where the host fell furthest behind — and the graph flattened it. That is the pattern to remember: at large batch, graphs buy you the tail even when they don't buy you the median.

## When capture breaks or the output is garbage

Now the failure modes, because you *will* hit them, and each one has a signature that points straight at the violated rule. This is the stress-test section: what happens when you forget a step, and how to read the symptom back to the cause.

![a decision tree that diagnoses a broken or garbage producing graph by asking whether warmup was skipped whether a fresh tensor is used each step and whether the shape or control flow changed](/imgs/blogs/cuda-graphs-in-pytorch-6.webp)

The tree walks the four questions in the order you should ask them, because they're ordered by how common the mistake is. Start at the top and take the first yes.

**You forgot the warmup.** Symptom: the capture itself hangs, or throws a CUDA error about an operation not permitted during capture, or the first replay errors. Cause: an un-warmed forward tries to autotune a cuBLAS/cuDNN kernel or grow the allocator *inside* the capture, both forbidden. This is the most common failure and the easiest to fix — add the warmup loop on the side stream and it clears. If it still hangs, the warmup shapes don't match the capture shapes, so autotuning fires again at capture time; make sure you warm up with the exact tensor you'll capture.

**You forgot to `copy_` — you bound a fresh tensor.** Symptom: no error at all, and *garbage output*, or worse, stale output that looks plausible because it's last step's answer. Cause: `g.replay()` read from the address it recorded at capture, but your new data went into a different tensor at a different address, so the graph never saw it. This is the nastiest failure because it's silent — the code runs, returns a tensor, and the tensor is wrong. The tell is that the output doesn't change (or changes only partially) when the input changes. Fix: always `static_in.copy_(new_data)`; never rebind. If you see a graphed model whose predictions are frozen or lagging, this is almost always why.

**The shape changed.** Symptom: an error on replay about a size mismatch, or — if the new shape happens to fit the recorded buffers — garbage, because the kernels were recorded for the old dimensions. Cause: a graph is one shape. A different sequence length, a different batch size, a different dtype: all need their own graph. Fix: bucket, as in the serving section, one graph per shape; or fall back to eager for off-bucket shapes.

**A control-flow op sneaked into the captured region.** Symptom: the capture errors, or the graph replays a *fixed* branch every time regardless of the data. Cause: anything data-dependent inside the capture — a Python `if x.item() > 0`, a `.cpu()`, a `print(tensor)`, an early-exit that depends on a tensor value — either forces a device-to-host sync (illegal during capture, because it would need to read a value the graph hasn't computed yet) or, if it's a Python-level branch evaluated once at capture time, gets *baked in* so replay always takes the branch it saw during capture. A graph is a straight-line recording; it has no `if`. Fix: move data-dependent control flow *outside* the captured region — capture the branch-free tensor math, and do the `if` in Python around the replay, choosing which graph to replay rather than branching inside one.

Here's the silent-garbage failure made concrete, because it's the one that costs the most debugging time:

```python
# WRONG: a fresh tensor every step. The graph never sees new_tokens.
@torch.no_grad()
def infer_broken(new_tokens):
    static_in = new_tokens.cuda()      # NEW allocation, NEW address
    g.replay()                         # replays against the OLD recorded address
    return static_out.clone()          # -> stale / garbage, no error raised

# RIGHT: copy INTO the recorded buffer.
@torch.no_grad()
def infer_ok(new_tokens):
    static_in.copy_(new_tokens)        # same address the graph recorded
    g.replay()
    return static_out.clone()
```

The two differ by one line and one produces confidently wrong answers with no exception. When a graphed model's outputs stop tracking its inputs, this is the first thing to check.

The control-flow failure has the same one-line character. A branch that reads a tensor value — early-exit on a stop token, a threshold on a logit, a length check — cannot live inside the captured region, because the branch is decided once, at capture time, and then frozen. The fix is to lift the decision *out* of the graph and let Python choose which graph to replay:

```python
# WRONG: the branch is baked in at capture; replay always takes the captured path.
with torch.cuda.graph(g):
    logits = model(static_in)
    if logits.max().item() > THRESH:   # .item() syncs -> illegal during capture
        out = cheap_head(logits)
    else:
        out = full_head(logits)

# RIGHT: capture the branch-free math; branch in Python around the replay.
with torch.cuda.graph(g_logits):
    static_logits = model(static_in)   # no data-dependent control flow inside

def infer(new_tokens):
    static_in.copy_(new_tokens)
    g_logits.replay()
    if static_logits.max().item() > THRESH:   # sync is fine OUTSIDE capture
        return cheap_head(static_logits)
    return full_head(static_logits)
```

A graph is a straight-line recording; it has no `if`, no `.item()`, no `print(tensor)`. Anything that needs to read a computed value goes outside the captured region. For the deeper catalogue — allocator interactions, capture invalidation, streams and events inside the region, the subtle ones — the [gotchas-and-debugging post](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) is the companion; this section is the fast triage.

## Verifying it worked: read the collapse off a trace

Never trust that graphing helped — *prove* it, in a trace, because it's entirely possible to capture a graph that doesn't actually replay (a shape mismatch silently falls back to eager) or to graph a region that wasn't your bottleneck. The proof is a before/after profile where the host-side launch calls collapse to a single graph launch per step. Use the same [torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) you'd reach for anywhere.

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_it(fn, batch, label):
    for _ in range(20):                          # warmup
        fn(batch)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn(batch)
            torch.cuda.synchronize()
    print(f"==== {label} ====")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=6))

profile_it(lambda b: model(b), batch, "eager")
profile_it(infer,               batch, "graphed")
```

The eager table shows what you already saw — a fat `cudaLaunchKernel` row with 18,000 calls over 10 steps. The graphed table is where you confirm the collapse: `cudaLaunchKernel` drops toward zero and a `cudaGraphLaunch` row appears with a count equal to your step count.

```console
==== graphed ====
--------------------------  ------------  ------------  ------------  ------------
                      Name    Self CPU %      Self CPU     Self CUDA    # of Calls
--------------------------  ------------  ------------  ------------  ------------
           cudaGraphLaunch        78.9%       2.84ms       0.000us            10
      cudaStreamSynchronize        14.1%       0.51ms       0.000us            10
               aten::copy_         3.8%       0.14ms       0.020ms            10
--------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 3.60ms
Self CUDA time total: 61.8ms
```

Three things confirm the win. First, `# of Calls` on the launch path is now **10** (`cudaGraphLaunch`, one per step) instead of 18,000 — the launches collapsed. Second, `Self CPU time total` fell from 120.4 ms to 3.6 ms — the host is no longer the bottleneck. Third, and this is the honesty check, `Self CUDA time total` is essentially unchanged, 62.1 ms then 61.8 ms — the device work didn't change, exactly as the mechanism predicted, which is how you *know* the graph replayed the same kernels rather than silently doing something different. If the graphed CUDA total were much *lower*, you'd suspect the graph skipped work; if the launch count didn't drop, you'd suspect a fallback to eager. The trace is the ground truth. On the Chrome timeline the same story shows as the GPU lane filling in — the wide idle gaps between kernels close up, because the host is no longer the pace-setter — which is the visual the [Chrome-trace post](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) teaches you to read.

## Case studies and real numbers

A few checkable results that calibrate what a CUDA-graph win looks like, so you know whether yours is in range.

**PyTorch's own CUDA-graphs work.** The PyTorch team's "Accelerating PyTorch with CUDA Graphs" blog documents the mechanism and the wins on launch-bound models, and reports that the largest gains come precisely from models built of many small kernels — the launch-bound signature. Their headline case, Mask R-CNN training, saw meaningful end-to-end step-time improvement from partial-network capture, and the deep-learning-examples repositories ship graphed training loops. The lesson that generalizes: the win scales with how host-bound you started, so measure your own $p$ (the profiler footer) rather than transplanting a headline multiplier.

**`make_graphed_callables` in NVIDIA's training recipes.** NVIDIA's optimized BERT and Transformer training configs use `make_graphed_callables` (or its Apex predecessor) to graph the fixed-shape transformer blocks while leaving variable-length input handling eager — the partial-graph pattern from the high-level-API section. The reported benefit is a reduction in per-step CPU overhead that matters most at small per-GPU batch, exactly where strong scaling pushes you. It's the canonical example of graphing *part* of a model rather than all of it.

**`reduce-overhead` on inference.** `torch.compile(mode="reduce-overhead")` composes Inductor fusion with CUDA graphs, and the PyTorch performance docs show it delivering the biggest relative wins on small-batch, launch-bound inference — again the same regime. The [compile-plus-graphs post](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead) walks how the two fixes compose and where the memory and dynamic-shape pitfalls are; the one-line summary is that it's the cheapest way to get both fixes when your shapes are fixed, and a recompilation liability when they aren't.

**The launch-overhead constant.** The ~5–10 µs per-launch host cost that makes all of this worth doing is not a PyTorch number — it's a CUDA driver characteristic, documented in NVIDIA's CUDA Graphs materials as the motivation for the graph API in the first place. It's why graphs exist: below roughly that threshold of device work per kernel, you spend more time launching than computing, and the only fixes are fewer launches (fusion) or one launch (a graph). The [model-serving kernel-fusion post](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) covers how fusion and graphs stack in a production LLM server.

## When to reach for CUDA graphs (and when not)

Graphs are a sharp tool with a real cost, and the discipline is knowing when the cost isn't worth it.

**Reach for them** when the profiler says launch-bound — Self CPU time dwarfing Self CUDA time, a fat `cudaLaunchKernel` row, high call counts on cheap ops — *and* your shapes are fixed or bucketable. That's the sweet spot: small-batch, latency-sensitive inference, or a training step with static shapes. The win is large, the accuracy cost is zero, and the code is a page.

**Don't reach for them when you're already device-bound.** If the profiler footer shows CPU and CUDA close, or CUDA exceeding CPU, your launch-overhead fraction $p$ is small, Amdahl caps your win near 1x, and the static-buffer bookkeeping buys you nothing but a maintenance burden and a class of silent-garbage bugs. Graphs don't make kernels faster; they only remove host overhead. If you're compute-bound at 90% occupancy, look at the kernel, not the launch path.

**Don't graph a service whose shapes change every request** without bucketing first. A graph is one shape; feed it a stream of varying sequence lengths and you either re-capture constantly (expensive, and a new failure surface) or fall back to eager on every miss (no win). Bucket the shapes, or use `torch.compile(dynamic=True)` and accept that dynamic shapes and graphs are partly at odds — the [serving-loop post](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) covers the padding-and-masking that makes variable input work with fixed graphs.

**Don't reach for the manual API when `make_graphed_callables` would do.** If you don't need to reach inside the captured region, the automated wrapper is safer — it handles the warmup handshake and the static buffers you'd otherwise get subtly wrong. Save the manual `torch.cuda.graph` for when you genuinely need control: a custom decode loop, a KV-cache update between sub-steps, a graph you compose with hand-written stream logic.

**Don't skip the verification.** A captured graph that silently falls back to eager, or that graphs the wrong region, is worse than no graph — you *think* you fixed it. Always confirm the launch collapse in a trace, and always check that the device CUDA total is unchanged (proof the same work ran) rather than lower (a sign work went missing).

## Key takeaways

- CUDA graphs in PyTorch are four stages: allocate static I/O once, warm up on a side stream, capture once, then `copy_` / `replay` / read every step. The first three are one-time; only the last repeats.
- The static-buffer discipline is the whole trick. A graph records raw device addresses, so you must `copy_` new data *into* the same input tensor and read *out of* the same output tensor. Bind a fresh tensor and replay silently returns garbage — no exception.
- Warmup is mandatory, not optional. It moves autotuning, allocator growth, and lazy init *out* of the capture region, where those operations are forbidden. Skip it and capture hangs or errors. Three to eleven iterations, on a side stream, with the exact shapes you'll capture.
- Graphing changes host cost, not device work. The device time per step is unchanged before and after; the win is collapsing N launches into one replay. That's the honesty check in your verification trace.
- Amdahl caps the win. If launch overhead is fraction $p$ of your step, the ceiling is $\frac{1}{1-p}$. Profile first: the CPU-versus-CUDA footer *is* your measurement of $p$. Big at small batch, small at large batch.
- Use `torch.cuda.graph` for control, `make_graphed_callables` for training steps (it captures backward too), and `torch.compile(mode="reduce-overhead")` for the one-line fusion-plus-graphs win on fixed-shape inference.
- A graph is one shape. Serve varying batch sizes by capturing one graph per bucket, padding requests up, and masking the padding. The biggest wins are at small batch; large batch mostly buys you the p99 tail.
- Graphs reserve a private memory pool up front and cannot allocate or free while replaying. Share a pool across buckets that never replay concurrently to save memory.
- Verify in a trace, always. Confirm `cudaLaunchKernel` collapsed to one `cudaGraphLaunch` per step and that the device CUDA total is unchanged. A graph you didn't verify is a fix you can't trust.

## Further reading

- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — the primary PyTorch blog: the mechanism, `make_graphed_callables`, and partial-network capture with measured wins.
- [PyTorch CUDA semantics: CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) — the reference for `torch.cuda.graph`, `CUDAGraph`, `graph_pool_handle`, warmup, and the static-buffer rules.
- [NVIDIA CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) — the driver-level API these PyTorch wrappers sit on, and the source of the launch-overhead motivation.
- [CUDA graphs from first principles](/blog/machine-learning/performance-engineering/cuda-graphs-from-first-principles) — the sibling that derives capture-versus-replay and why the recorded DAG eliminates per-launch cost.
- [The kernel launch overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — where the launch-bound signature and the $N \cdot t_\text{launch}$ accounting come from.
- [CUDA graphs gotchas and debugging](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) — the deeper failure catalogue: capture invalidation, allocator interactions, streams inside the region.
- [CUDA graphs in a serving loop](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) — bucketing, per-shape graphs, and combining graphs with continuous batching in a real service.
- [Compile plus CUDA graphs: reduce-overhead](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead) — how `torch.compile` composes fusion with graphing, and the dynamic-shape pitfalls.
- [Profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) — how to read the footer that tells you whether graphing will even help.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree: symptom to tool to cause to fix, with CUDA graphs as the launch-bound answer.
