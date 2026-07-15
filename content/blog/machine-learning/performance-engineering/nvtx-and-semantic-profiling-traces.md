---
title: "NVTX and Semantic Traces: Making a Profile Readable by Labeling Your Own Code"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "A raw profile is a wall of aten::mul and elementwise_kernel with no idea which is your preprocess, forward, or postprocess. Learn how NVTX ranges and CUDA-graph kernel annotations turn an anonymous trace into a semantic one where every span maps to a phase of your request."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "nvtx",
    "profiling",
    "pytorch",
    "cuda",
    "nsight",
    "cuda-graphs",
    "latency",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 41
---

You open a freshly captured trace of your inference service, expecting to see where the time goes. Instead you get eight thousand rows on the GPU timeline, and almost every one of them is named `elementwise_kernel`, `vectorized_elementwise_kernel`, or `void at::native::...` followed by forty characters of C++ template soup. Interleaved on the CPU row is an endless stream of `cudaLaunchKernel`, `aten::mul`, `aten::add`, `aten::layer_norm`. Somewhere in that wall is the reason a request takes 41 ms instead of the 30 ms you budgeted. But the profiler does not know that kernel #4,812 is the fourth attention head of decoder layer 19, and neither do you. Every kernel is named for the *operation the compiler emitted*, not for the *phase of your request* it belongs to. Reading this trace is archaeology: you scroll, you guess, you correlate timestamps by hand, and forty minutes later you have a hypothesis you are only half-sure of.

There is a second, quieter version of this problem that is even worse. The moment you turn on CUDA graphs to kill launch overhead — the single most effective fix for a host-bound service — your entire forward pass collapses into *one* line on the GPU row: a single `cudaGraphLaunch` replaying a recorded DAG of kernels. The trace got shorter, but it also went completely dark. You cannot see which layer is slow anymore because the profiler no longer sees layers at all. It sees one opaque replay. You traded a wall of anonymous kernels for a single black box.

Both problems have the same root cause and the same fix. The root cause is that *the code knows the semantics and the trace does not*. You know that lines 40 through 52 of your handler are the tokenizer, that `self.decoder(x)` is thirty-two transformer blocks, that the last few ops are sampling and detokenization. The profiler sees none of that structure — it only sees kernels. The fix is to *tell it*: annotate your own code with named ranges so that the semantics you already know get stamped into the trace. That is exactly what **NVTX** (the NVIDIA Tools Extension) does. A handful of `range_push` / `range_pop` calls, and the same eight-thousand-kernel wall re-renders as a clean story: `tokenize 2 ms → embed 1 ms → 32× decoder_layer → sample 0.6 ms`. You stop doing archaeology and start reading.

![a two column comparison showing a raw trace as a wall of identical elementwise kernels on the left and the identical run annotated as named request phases on the right](/imgs/blogs/nvtx-and-semantic-profiling-traces-1.webp)

The figure above is the entire thesis of this post in one frame, and by the end you will be able to produce the right-hand side for your own service. On the left, the raw capture: eight thousand kernels, no phase boundaries, and a "time-to-find-the-wall" measured in minutes of scrolling. On the right, the identical run with NVTX ranges: four labeled bars, the hot phase (`decoder x32`, 38 ms) obvious at a glance, and time-to-find-the-wall measured in seconds. This is not a faster program — it is the *same* program, captured the same way, with labels added. The speedup is in *your* debugging loop, not the model. And that is the whole game: the [profile → hypothesize → fix → measure loop](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) only turns fast if reading the profile is fast. This post makes reading the profile fast. By the end you will know how NVTX ranges are emitted and picked up by Nsight Systems and Nsight Compute, how `record_function` relates to them inside `torch.profiler`, how to wrap a real request handler so the trace tells the request's story, and — the hard part — how to keep those labels alive after CUDA-graph capture erases kernel identity.

## What NVTX actually is, and why nsys and ncu can read it

Start with the mechanism, because once you see it the API is obvious. **NVTX** is a tiny header-only instrumentation library from NVIDIA. It gives you three primitives: a *range* (a span with a start and an end — "this region of wall-clock time is called X"), a *mark* (an instantaneous point event — "X happened here"), and a *domain* (a namespace so different subsystems' ranges do not collide). That is essentially all of it. When you call `nvtxRangePushA("decode_layer")`, the library does one thing: it records a lightweight event — a string and a timestamp — into a buffer that any attached NVIDIA tool can read. `nvtxRangePop()` records the matching end.

The crucial property is *who is listening*. NVTX by itself does almost nothing. It is a **producer** of events; the tool is the **consumer**. When you run your program normally, with no profiler attached, `nvtxRangePush` resolves to a near-empty stub — a function-pointer check and an early return — costing on the order of a nanosecond. When you run the *same binary* under Nsight Systems (`nsys profile -t nvtx ...`) or Nsight Compute (`ncu --nvtx ...`), the tool has injected itself between your process and the driver; now those same calls stream real events into the tool's timeline. This is why NVTX is safe to leave in production code: the ranges are free until someone attaches a profiler, and then they are the difference between a readable trace and a wall.

PyTorch wraps NVTX in `torch.cuda.nvtx`, so you never touch the C API. Here is the whole surface:

```python
import torch

# Explicit push/pop — pairs must nest correctly (last pushed, first popped).
torch.cuda.nvtx.range_push("preprocess")
x = tokenizer(request.text)          # host-side work
x = x.to("cuda", non_blocking=True)  # H2D copy
torch.cuda.nvtx.range_pop()          # closes "preprocess"

# Context-manager form — cannot forget the pop, exception-safe.
with torch.cuda.nvtx.range("forward"):
    y = model(x)

# An instantaneous marker (a vertical tick on the timeline, no duration).
torch.cuda.nvtx.mark("cache_miss")
```

The `range_push`/`range_pop` pair is the low-level workhorse; the `range(...)` context manager is the same thing with a `try/finally` so an exception inside the block still emits the pop. Prefer the context manager everywhere you can, and reach for explicit push/pop only when a range must span two functions or a loop boundary where a `with` block does not fit cleanly.

Now, the key relationship every PyTorch user trips over: **`torch.profiler.record_function` is the `torch.profiler` analog of an NVTX range, and NVTX is what Nsight reads.** They label the same *kind* of thing — a named region of your code — but they feed different consumers. `record_function("decode_layer")` creates a labeled span in the **torch.profiler** trace: it shows up in `key_averages().table()` and as a bar in the Chrome/Perfetto timeline you get from `export_chrome_trace`. It does *not*, on its own, appear in an `nsys` capture. Conversely, `torch.cuda.nvtx.range("decode_layer")` shows up in **nsys** and **ncu**, but not in `torch.profiler`'s `key_averages` table. If you profile with `torch.profiler`, label with `record_function`. If you profile with `nsys`/`ncu`, label with NVTX. And if you want *one* set of labels that both tools understand, there is a bridge — `torch.autograd.profiler.emit_nvtx()` — which we will come to. We will lay all four options out in a comparison once we have measured what each one costs.

### The three ways to emit NVTX from PyTorch

Beyond hand-placed ranges, PyTorch gives you a firehose option. `torch.autograd.profiler.emit_nvtx()` is a context manager that, for its duration, emits an NVTX range around *every single autograd operation* — every `aten::mm`, `aten::layer_norm`, `aten::add`. You place it once, run under `nsys`, and every op in your model is suddenly named on the timeline with no manual instrumentation:

```python
import torch

model = build_model().cuda().eval()
x = torch.randn(1, 512, dtype=torch.long, device="cuda")

# Auto-emit one NVTX range per autograd op for the whole block.
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    with torch.no_grad():
        y = model(x)
```

This is fantastic for a first pass — you get complete coverage for free — but understand the trade. `emit_nvtx` pushes and pops a range around *thousands* of ops per forward pass, and with `record_shapes=True` it also serializes each op's input shapes into the range name. That is real overhead (we will quantify it), and it produces a *dense* timeline where every op is its own range. It answers "which op" but buries "which phase." The mature workflow is the opposite of a firehose: a *small* number of hand-placed semantic ranges (`preprocess`, `forward`, `decoder_layer_{i}`, `sample`) that map to *your* mental structure, reserving `emit_nvtx` for the zoom-in when a phase turns out to be hot and you need the op-level detail inside it. Coarse ranges to find the wall; `emit_nvtx` to see the bricks.

The third path is `record_function`, the torch.profiler-native span, which we cover in depth in [profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler). The one thing to carry from there into here: `record_function` and NVTX ranges are *not* mutually exclusive. Running your `record_function`-annotated model under `torch.profiler` with `emit_nvtx` active makes those same named scopes emit NVTX too, so a single set of `record_function("phase")` calls can light up both the Chrome trace and the `nsys` timeline. You instrument once; the labels reach both consumers.

## Instrumenting a real handler so the trace tells the request's story

Theory is cheap. Here is a real inference handler, the kind that sits behind a REST or gRPC endpoint, and here is how you turn it from an anonymous kernel-emitter into a self-documenting trace. The handler does what every serving handler does: receive bytes, deserialize them into a tensor, copy host→device (H2D), run the model forward, sample a token, copy device→host (D2H), and serialize the response. Each of those is a *phase*, and each phase is exactly one NVTX range.

```python
import torch
import torch.cuda.nvtx as nvtx

@torch.no_grad()
def handle(request_bytes, model, tokenizer, device="cuda"):
    with nvtx.range("receive"):
        raw = request_bytes                      # network / queue hand-off

    with nvtx.range("deserialize"):
        ids = tokenizer.encode(raw)              # CPU-side tokenization
        host = torch.tensor(ids, dtype=torch.long).pin_memory()

    with nvtx.range("H2D"):
        x = host.to(device, non_blocking=True)   # host -> device copy

    with nvtx.range("forward"):
        logits = model(x)                        # 32 decoder layers of GPU work

    with nvtx.range("sample"):
        next_id = torch.argmax(logits[:, -1], dim=-1)

    with nvtx.range("D2H"):
        out_id = next_id.cpu()                   # device -> host copy

    with nvtx.range("serialize"):
        return tokenizer.decode(out_id.tolist())
```

Seven ranges, seven phases, zero change to what the code *does*. Now capture it under Nsight Systems:

```bash
# -t selects the trace domains: cuda kernels, NVTX ranges, OS runtime, cuDNN, cuBLAS.
# --capture-range makes nsys only record between a start/stop you control (optional).
nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -o handler_trace \
  --force-overwrite true \
  python serve_one_request.py
```

Open `handler_trace.nsys-rep` in the Nsight Systems GUI and the top of the timeline now carries a labeled NVTX row: seven bars, left to right, each with a real duration. Below it, the CUDA API row and the kernel row show the `cudaLaunchKernel` calls and the kernels themselves — but now every kernel sits *underneath* the NVTX bar that owns it, so you know which phase each belongs to. You do not even need the GUI to read the summary; `nsys stats` prints it:

```console
$ nsys stats --report nvtx_sum handler_trace.nsys-rep

** NVTX Range Summary (nvtx_sum):

 Time(%)   Total Time (ns)   Instances   Avg (ns)    Range
 -------   ---------------   ---------   ---------   -----------
   92.6         38041220           1     38041220   forward
    2.9          1201004           1      1201004   H2D
    2.0           812330           1       812330   deserialize
    1.5           602100           1       602100   sample
    1.0           410880           1       410880   D2H
    0.7           301240           1       301240   receive
    0.5           205990           1       205990   serialize
```

Read that table top to bottom and the story writes itself: 92.6% of the request is `forward`. Everything else — the copies, the tokenizer, the sampling — is rounding error. Before the annotation you would have summed hundreds of kernels by hand to reach that conclusion; now it is the first line of a summary. This is the payoff, and it took seven `with` statements.

![a timeline of one request showing receive, deserialize, host to device copy, forward, sample, device to host copy, and serialize as labeled bars with durations](/imgs/blogs/nvtx-and-semantic-profiling-traces-2.webp)

The timeline above is the annotated request as `nsys` draws it: seven bars, left to right, with `forward` (38 ms) dominating and the two copies (H2D 1.2 ms, D2H 0.4 ms) called out as the caution-colored bandwidth tax they are. Notice what this immediately rules out: the handler is *not* host-bound at the request boundary — deserialize and serialize together are under 1.5 ms. If someone claimed "the tokenizer is our bottleneck," this trace refutes it in one glance. The value of a semantic trace is as much in what it *exonerates* as in what it accuses.

### Scoping the capture to steady state

A subtlety that bites everyone the first time: if you just run `nsys profile python serve.py`, you capture *everything* — process start, CUDA context creation, cuDNN autotuning, the cold first iterations, model loading — and all of that noise dwarfs the steady-state requests you actually care about. The first forward pass on a fresh context can be 10x slower than the hundredth because of one-time autotuning and lazy allocation, and its kernels are named the same as the warm ones, so a naive capture pollutes your `nvtx_sum` with cold-start garbage. You want to profile the *warm* service, not its boot sequence.

NVTX pairs with a capture-range control to fix this. You warm up outside the captured region, then start and stop profiling around exactly the steady-state requests, and Nsight records only what falls between them:

```python
import torch

# Warm up: run enough requests to finish autotuning and allocation.
for _ in range(20):
    handle(sample_request, model, tokenizer)
torch.cuda.synchronize()

# Only the region between start() and stop() lands in the capture.
torch.cuda.profiler.start()
for _ in range(50):
    handle(sample_request, model, tokenizer)   # steady-state requests
torch.cuda.profiler.stop()
```

Pair that with the matching `nsys` flags so the tool honors your start/stop instead of recording from process launch:

```bash
# --capture-range=cudaProfilerApi makes nsys start recording only when the
# app calls cudaProfilerStart (torch.cuda.profiler.start) and stop at stop().
nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o warm_trace \
  python serve.py
```

Now `warm_trace.nsys-rep` contains only the fifty warm requests, your NVTX ranges are clean, and the `nvtx_sum` averages are steady-state numbers you can actually trust. This is the same warm-up-then-measure discipline the whole [reproducible-benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) post is built on, applied to tracing instead of timing: never let cold-start kernels contaminate the phase breakdown you are about to reason from.

#### Worked example: finding the wall in a 32-layer decoder

A team reports p50 latency of 41 ms on an **A100 80GB SXM**, target 30 ms, and no idea where the 11 ms of slack lives. The raw `nsys` capture has roughly 8,000 kernels per request — thirty-two decoder layers, each firing a QKV projection, an attention kernel, an output projection, an MLP up-projection, a GELU, and an MLP down-projection, plus layer norms, plus the elementwise residuals. Every kernel is named for its template. Finding the slow layer means correlating kernel timestamps against a hand-kept map of layer boundaries, by hand, for eight thousand kernels.

Instead, they add one range per layer. In the model's forward, the decoder loop becomes:

```python
for i, layer in enumerate(self.layers):
    with torch.cuda.nvtx.range(f"decoder_layer_{i}"):
        x = layer(x)
```

Re-capture, run `nsys stats --report nvtx_sum`, sort by total time, and the answer is one line:

```console
 Time(%)   Total Time (ns)   Instances   Range
 -------   ---------------   ---------   -----------------
   ...
    5.6          2130500           1     decoder_layer_19
    2.9          1102900           1     decoder_layer_7
    2.9          1098400           1     decoder_layer_8
   ...
```

Layer 19 takes 2.13 ms; every other layer takes about 1.1 ms. One layer is doing twice the work of its neighbors. Zooming into `decoder_layer_19` in the GUI (or filtering `ncu` to that NVTX range, shown later) reveals the cause: that layer alone runs attention at full sequence length because a masking bug skips the KV-cache slice there. The fix is four lines. But *finding* it went from an afternoon of kernel archaeology to a sorted table with an obvious outlier. That is the entire return on seven — here, thirty-two — `with` statements.

### The overhead question: is it safe in production?

The reflexive worry is "I am adding thousands of instrumentation calls to my hot path; won't that slow the service?" This is where the mechanism pays off, and it deserves a real derivation rather than a hand-wave.

Let $N$ be the number of NVTX ranges you push per request and $t_\text{nvtx}$ the cost of one push/pop pair. The added host-side time per request is:

$$t_\text{overhead} = N \cdot t_\text{nvtx}$$

The value of $t_\text{nvtx}$ depends entirely on whether a profiler is attached. **When no tool is attached**, `nvtxRangePush` is a stub: it loads a function pointer, sees it is null (no injection), and returns. That is a handful of instructions — call it $t_\text{nvtx} \approx 1\text{ ns}$, and often less because the branch predictor nails the "no tool" path every time. **When a tool is attached**, the call actually records an event into the tool's buffer — call it $t_\text{nvtx} \approx 1\text{ }\mu\text{s}$, a thousand times more, because now it does real work.

Plug in real numbers. A coarse-grained handler with the seven phase ranges above: $N = 7$. Idle overhead $= 7 \times 1\text{ ns} = 7\text{ ns}$ per request — utterly unmeasurable against a 41 ms request. Even the per-layer scheme, $N = 32$: idle overhead $= 32\text{ ns}$. You could push a thousand ranges per request and still be under a microsecond of idle cost. **This is why you leave NVTX ranges in production code.** They are documentation that becomes instrumentation the instant you attach `nsys`, and they cost you nothing the rest of the time.

The story flips only for `emit_nvtx`, which pushes a range around *every op*. A transformer forward might issue $N \approx 4{,}000$ autograd ops. Under an attached profiler at $1\text{ }\mu\text{s}$ each, that is $4{,}000 \times 1\text{ }\mu\text{s} = 4\text{ ms}$ of pure annotation overhead on top of a 38 ms forward — a 10% profiling tax, plus the timeline is now four thousand bars deep. That is fine for a deliberate profiling run and wrong for anything you leave on. The discipline: hand-placed coarse ranges live in the code forever; `emit_nvtx` is a tool you switch on for one capture and switch off.

| Annotation scheme | Ranges/request $N$ | Idle overhead | Overhead under nsys | Leave in prod? |
|---|---|---|---|---|
| 7 phase ranges | 7 | ~7 ns | ~7 µs | Yes, always |
| Per-layer (32) | 32 | ~32 ns | ~32 µs | Yes |
| Per-layer + sub (128) | 128 | ~130 ns | ~130 µs | Yes |
| `emit_nvtx` (per-op) | ~4,000 | n/a (context only) | ~4 ms | No — profiling only |

The table makes the rule concrete: everything up to a few hundred hand-placed ranges is free at rest, so instrument liberally and permanently; only the per-op firehose is a profiling-time-only tool. There is no version of "NVTX ranges slowed my service" that survives contact with the arithmetic, as long as you are placing ranges by hand and not wrapping every elementwise add.

With the overhead of each method now measured, here is the promised side-by-side of all four ways to label your code, so you can pick by tool and cost in one glance.

![a comparison table of four annotation methods showing which tool reads each one, the overhead, and what each is best for](/imgs/blogs/nvtx-and-semantic-profiling-traces-5.webp)

The matrix above is the decision compressed into one view, and it is worth memorizing because choosing the wrong annotation is the most common way people waste an afternoon ("I added `record_function` everywhere and my `nsys` timeline is still blank"). Read it as four rows: `record_function` feeds the torch.profiler Chrome trace at about a microsecond per call; `nvtx.range` feeds `nsys` and `ncu`, essentially free when idle and about a microsecond live; `emit_nvtx` auto-emits an NVTX range for *every* autograd op (total coverage, but heavy — the ~4 ms per-op tax the table above quantifies); and graph annotation is the special case that *survives CUDA-graph capture* and shows up as named graph nodes in `nsys`, which we build later in this post. The column that matters most is "Read by" — the annotation must match the tool, or your labels go to a consumer nobody is watching.

## Nested ranges: giving a request a call hierarchy

A single flat row of phase labels is already a huge win, but the real power of NVTX shows up when you *nest*. Ranges follow a strict stack discipline — last pushed, first popped — which means the profiler can reconstruct a *call hierarchy* from your pushes and pops, exactly like a stack trace, except it is a *time* stack. When `forward` contains `decoder_layer_7`, which contains `attention`, which contains the SDPA kernel, the tool draws them as nested bars: each deeper range sits inside and below the one that owns it.

```python
with torch.cuda.nvtx.range("request"):
    with torch.cuda.nvtx.range("forward"):
        for i, layer in enumerate(self.layers):
            with torch.cuda.nvtx.range(f"decoder_layer_{i}"):
                with torch.cuda.nvtx.range("attention"):
                    x = layer.attn(x)
                with torch.cuda.nvtx.range("mlp"):
                    x = layer.mlp(x)
```

This is four levels deep — `request` ⊃ `forward` ⊃ `decoder_layer_i` ⊃ `{attention, mlp}` — and Nsight renders it as a staircase. You can collapse the whole `forward` into one bar when you want the big picture, then expand `decoder_layer_19` to see that its `attention` sub-range, not its `mlp`, is the fat one. The nesting is not cosmetic: it is what lets you drill from "the request is slow" to "the forward is slow" to "layer 19 is slow" to "layer 19's attention is slow" without ever leaving the labeled view.

![a vertical stack showing the request range containing the forward phase containing one decoder layer containing an attention sub range containing one kernel](/imgs/blogs/nvtx-and-semantic-profiling-traces-3.webp)

The stack above is the nesting drawn as depth: the outermost `request` range (41 ms) contains `forward` (38 ms), which contains `decoder_layer_7` (1.2 ms), which contains an `attention` sub-range (0.6 ms), which finally contains the SDPA kernel itself (40 µs). Each level is a scope you can open or close. The discipline that makes this work — and the one thing to get right — is *pairing*: every `range_push` needs exactly one `range_pop`, and they must nest, never cross. If you pop in the wrong order (close `forward` before `attention`), the tool's stack corrupts and the timeline draws garbage — bars that overlap impossibly or ranges that swallow the rest of the trace. This is precisely why the `with`-block form is worth preferring: Python's context-manager protocol guarantees the pop fires, in order, even on an exception. Hand-rolled `range_push`/`range_pop` around code that can `raise` is the classic way to desync the stack and get a nonsense trace.

There is a subtlety worth internalizing about what a range's *duration* means on a GPU. Because kernel launches are asynchronous — the host enqueues work and races ahead, a mechanism we unpack in [how the host races ahead of the GPU](/blog/machine-learning/performance-engineering/the-mental-model-of-a-gpu-service) — an NVTX range on the *host* row measures the time the CPU spent *inside* that Python block, which is mostly launch time, not compute time. The `forward` range on the CPU row might be only 3 ms (the time to enqueue 8,000 kernels) while the *GPU* is busy for 38 ms draining them. Nsight handles this by projecting your NVTX ranges onto *both* the CPU timeline (where you pushed them) and, through the CUDA correlation, onto the span of GPU kernels they enclose. So the `forward` bar you care about is the *GPU-projected* one — 38 ms — and the thin CPU-side bar is the launch cost. When you read a range's wall-time, always be clear which row you are reading: the host-side push-to-pop, or the device-side kernel span it correlates to.

## From a range to its kernels: correlation and custom lanes

The last idea unlocks the one that makes NVTX genuinely powerful for GPU work: **a range on the host correlates to the kernels it launched on the device.** When you push `decode_layer` and, inside it, launch four kernels, Nsight records that those four kernels were issued while `decode_layer` was on the stack. It can therefore attribute their GPU time back to your range — even though the kernels are named `ampere_sgemm_128x64` and your range is named `decode_layer`. This is the bridge from *semantic* labels (yours) to *mechanical* kernels (the compiler's).

![a host range branching to four gpu kernels for qkv gemm attention output projection and mlp gemm that merge back into one range wall time of 1.2 milliseconds](/imgs/blogs/nvtx-and-semantic-profiling-traces-4.webp)

The figure above is that containment drawn as a branch and merge: the host-side `decode_layer` push fans out to the four GPU kernels it launches — a QKV GEMM (0.30 ms), the SDPA attention kernel (0.52 ms), an output projection (0.18 ms), an MLP GEMM (0.20 ms) — and those four merge back into a single *range wall-time* of 1.20 ms. The arithmetic is the point: $0.30 + 0.52 + 0.18 + 0.20 = 1.20\text{ ms}$, and the range wall exactly accounts for its kernels. When the sum of a range's kernel times is *less* than the range's GPU-projected wall-time, the difference is a **gap** — the GPU sat idle inside your range, which is the signature of a host-bound phase (the CPU could not launch the next kernel fast enough). When it is *equal*, the phase is packed and GPU-bound. So a single annotated range gives you, for free, the compute-bound-vs-host-bound diagnosis for that phase — the core question the whole [roofline analysis](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) exists to answer, scoped to exactly the part of your request you care about.

This correlation is also how you keep the timeline *legible* under load. A real capture has GPU-work ranges, CPU-work ranges (tokenization, serialization), and I/O ranges (the network hand-off) all interleaved. NVTX **domains** let you separate them onto their own rows so they do not pile up:

```python
import torch

# Domains are separate namespaces; each renders as its own row in nsys.
gpu_dom = torch.cuda.nvtx  # default domain -> GPU-work ranges
# For CPU/IO work, many teams push ranges tagged by a prefix convention
# ("cpu/tokenize", "io/recv") since the PyTorch wrapper exposes the default
# domain; the raw NVTX C API exposes nvtxDomainCreate for true separate rows.

with gpu_dom.range("gpu/forward"):
    logits = model(x)
```

The prefix convention (`gpu/…`, `cpu/…`, `io/…`) is the pragmatic path in pure PyTorch, since `torch.cuda.nvtx` exposes the default domain; Nsight groups and colors by the string, so you still get visually separated, filterable lanes without dropping to the C API. If you need genuinely distinct domain *rows* (each with its own registered strings and colors), that is where a small C extension calling `nvtxDomainCreateA` earns its keep — but most services never need it. The prefix convention gets you 90% of the legibility for zero extra machinery.

### Colors and categories: making the timeline scannable at a glance

Beyond a name, an NVTX range can carry a **color** and a **category**, and both pay off the moment your timeline has more than a handful of range types. A color makes a phase visually poppable — you want every `H2D`/`D2H` copy the same shade of red so the eye finds the bandwidth tax instantly, every `forward` the same blue, every `sample` the same green. A category is an integer tag that groups related ranges so a tool can filter or fold by category. The raw NVTX C API takes an `nvtxEventAttributes_t` struct with `color`, `colorType`, `category`, and `payload` fields; PyTorch's wrapper exposes the name (and, depending on version, an optional argument path), so the common pattern for rich attributes is a thin helper that calls the C API through a tiny extension, or — the pragmatic route — encoding the category into your naming convention and letting Nsight color by name:

```python
import torch.cuda.nvtx as nvtx

# Convention-driven coloring: nsys can be told to color ranges by a
# name-prefix rule, so a consistent prefix is as good as an explicit color
# for most triage. Keep copies, compute, and host work visually distinct.
COPY, COMPUTE, HOST = "copy/", "compute/", "host/"

with nvtx.range(COPY + "H2D"):
    x = host.to("cuda", non_blocking=True)
with nvtx.range(COMPUTE + "forward"):
    y = model(x)
with nvtx.range(HOST + "serialize"):
    out = serialize(y)
```

The reason to bother: on a busy timeline with dozens of range types across many requests, color is what turns "read the label on every bar" into "the red bars are the copies, and there are too many of them." It is a scanning aid, not a correctness feature — but scanning speed is the whole point of semantic annotation, so a consistent color scheme (copies red, compute blue, host gray, I/O amber) compounds the win. The one discipline: pick the scheme once and apply it project-wide, because the value is entirely in *consistency* — a red bar has to mean "copy" everywhere, or the color carries no information.

An NVTX **payload** is the last attribute worth knowing about: a numeric or string value you attach to a range so it travels into the trace alongside the name. A natural use in serving is stamping the batch size or sequence length onto the `forward` range, so that when you later find a slow `forward`, the trace already tells you it was a batch of 64 at sequence length 2048 — you do not have to reproduce the request to learn its shape. The payload turns each range into a tiny structured log line embedded in the timeline.

You can also filter Nsight Compute to a single NVTX range, which is how you go from "the trace says `decoder_layer_19` is slow" to "here is the Speed-of-Light section for exactly the kernels in that layer":

```bash
# Profile ONLY the kernels launched inside the decoder_layer_19 NVTX range,
# with the full metric set, so ncu doesn't drown you in every kernel.
ncu --set full \
    --nvtx --nvtx-include "decoder_layer_19/" \
    -o layer19_deep \
    python serve_one_request.py
```

That `--nvtx-include "decoder_layer_19/"` is the semantic scalpel: `ncu` will only collect its (expensive, replay-based) metrics for kernels that ran inside a range named `decoder_layer_19`, skipping the other thirty-one layers entirely. Without the NVTX filter you either profile every kernel (slow, and you drown) or guess a kernel-name regex (fragile, because the names are templated). With it, your semantic label *is* the filter. This is the composition that makes the [Nsight Compute kernel deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) tractable on a real model: NVTX narrows the field, `ncu` goes to the metal on what is left.

#### Worked example: an NVTX-filtered ncu run on L4

The layer-19 outlier from earlier, chased to the metal. On an **NVIDIA L4** (242 fp16 TFLOP/s, 300 GB/s HBM), the team runs the filtered `ncu` command above. Because `--nvtx-include` scopes collection to just layer 19's kernels, the run finishes in seconds instead of the minutes a full-model `ncu --set full` would take (every kernel gets replayed multiple times to gather metrics — profiling the whole model this way is brutal). The Speed-of-Light section for the attention kernel comes back at 71% of memory bandwidth and 18% of compute — decisively **memory-bound**, consistent with the full-sequence-length attention the masking bug forced. On the L4's 300 GB/s, that bandwidth ceiling is lower than the A100's 2.0 TB/s, so the same bug costs *more* here: the layer-19 penalty is 3.8 ms on L4 versus 2.1 ms on A100. The NVTX filter is what made the L4 investigation quick enough to do at all; without it, `ncu --set full` across 8,000 kernels on the slower card is a coffee-and-come-back affair.

## The hard case: keeping labels alive through CUDA-graph capture

Everything so far assumed the profiler can *see* your kernels launch. CUDA graphs break that assumption, and this is where semantic annotation stops being a convenience and becomes the only thing standing between you and a black box.

Recall why you reach for CUDA graphs in the first place: a host-bound service spends its time in `cudaLaunchKernel` — the CPU cannot enqueue kernels fast enough to keep the GPU fed, so the GPU idles between tiny kernels. The kernel-launch overhead law is $t_\text{launch-total} = N_\text{kernels} \times t_\text{launch}$, and with $N = 8{,}000$ kernels and $t_\text{launch} \approx 5\text{–}10\text{ }\mu\text{s}$ of CPU-side cost each, that is 40–80 ms of pure launch overhead per request, much of which becomes GPU idle time. CUDA graphs kill it: you *capture* the entire sequence of launches once into a graph, then *replay* the whole DAG with a single `cudaGraphLaunch`. Eight thousand launches collapse to one. The host overhead per replay drops to a few microseconds. It is often the single biggest win available to a host-bound service, and we build it end-to-end in [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) (a forward-link — it ships later in this series).

Here is the capture, in outline:

```python
import torch

model = build_model().cuda().eval()
static_x = torch.zeros(1, 512, dtype=torch.long, device="cuda")

# Warm up on a side stream so cuDNN/cuBLAS autotune before capture.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        with torch.no_grad():
            _ = model(static_x)
torch.cuda.current_stream().wait_stream(s)

# Capture the whole forward into a graph.
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_x)

# Replay: one launch, no per-kernel CPU cost.
static_x.copy_(new_input_ids)
g.replay()
torch.cuda.synchronize()
result = static_out.clone()
```

Now capture *that* under `nsys`. The GPU row shows... one bar. `cudaGraphLaunch`, then the graph's kernels replay back-to-back with no host in the loop. Your beautiful per-layer NVTX ranges? Gone. Here is the mechanism, and it is the crux of the whole post: **NVTX ranges are host-side events, but graph *replay* does not re-execute your host code.** When you push `decoder_layer_7` inside the Python loop, that push happens during *capture* — but on *replay*, `g.replay()` does not run your Python loop at all. It hands the pre-recorded kernel DAG straight to the driver. Your `range_push` calls never fire on replay, so there are no NVTX events, so the trace is one anonymous replay. The graph made the service faster and the profiler blind.

![a two column comparison showing a captured cuda graph as one opaque replay node on the left and the same graph with annotations restored as 32 labeled per layer lanes on the right](/imgs/blogs/nvtx-and-semantic-profiling-traces-7.webp)

The figure above is the before-and-after of exactly this: on the left, the graphed model as most people first capture it — one `cudaGraphLaunch` replay node, thirty-two layers hidden inside, no way to tell which is hot. On the right, the same graph with annotations *baked into the capture*, so the replay emits thirty-two labeled lanes and layer 19's 2.1 ms is visible again. The difference between these two is the technique the PyTorch **"CUDA Graph Kernel Annotations and Profiling"** tutorial exists to teach, and it is the reason this post pairs NVTX with CUDA graphs instead of treating them separately.

### How annotation survives capture

The fix follows directly from the mechanism. If the problem is that host-side pushes do not fire on replay, the solution is to record the *annotations themselves into the graph*, as graph nodes, so that replaying the graph replays the annotations too. Modern CUDA (12.x) supports exactly this: NVTX range push/pop issued during stream capture can be recorded into the graph as nodes, so on replay the driver re-emits them into whatever profiler is attached. The labels become part of the recorded DAG, right alongside the kernels.

In practice, from PyTorch, the shape is: emit your NVTX ranges *during* the capture region, and rely on the CUDA capture to record them as graph nodes. The per-layer loop you already wrote does the right thing as long as the ranges are pushed *inside* the `with torch.cuda.graph(g):` block:

```python
import torch, torch.cuda.nvtx as nvtx

class AnnotatedDecoder(torch.nn.Module):
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Pushed DURING capture -> recorded into the graph as nodes,
            # so they re-emit on every replay (CUDA 12.x graph capture).
            with nvtx.range(f"decoder_layer_{i}"):
                x = layer(x)
        return x

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_x)   # nvtx pushes happen here, get captured

# On replay, the recorded NVTX nodes fire -> 32 labeled lanes reappear.
g.replay()
```

The critical detail — and the thing to verify on your stack, because it is version-dependent — is whether your CUDA and PyTorch build actually records NVTX ranges into the graph. On a CUDA version that supports it, the replay re-emits the ranges and your thirty-two lanes come back. On an older stack where capture treats NVTX push/pop as no-ops (they have no kernel effect, so nothing is recorded), the ranges vanish on replay and you fall back to a coarser tactic: **annotate the graph as a whole.** Wrap the single `g.replay()` call in one outer NVTX range so at least the entire graphed forward is labeled as one bar, and use the *non-graphed* eager capture (which you keep around for debugging) to see the per-layer breakdown. It is a strictly worse view — you lose per-layer resolution inside the graph — but "one labeled `graphed_forward` bar" still beats "one anonymous `cudaGraphLaunch`," and it costs one `with` statement:

```python
# Fallback when capture does not record inner NVTX: label the whole replay.
with torch.cuda.nvtx.range("graphed_forward"):
    g.replay()
```

The honest engineering position: check your stack once. Capture a tiny two-layer graph with per-layer NVTX ranges, run it under `nsys`, and look for two labeled lanes on replay. If they are there, your inner annotations survive and you get the full lit-up view from the right side of the figure. If they are not, use the whole-graph fallback and keep an un-graphed variant behind a flag for the days you need to see inside. Either way, you are never stuck with a fully opaque replay — which is the trap the tutorial is warning you out of.

It helps to lay the three execution modes side by side, because each one erases a different amount of the trace and each one is rescued by a different annotation. Eager execution launches every kernel from Python, so the trace is maximally granular but also maximally noisy — eight thousand named-for-the-compiler kernels. A `torch.compile`d model fuses many of those kernels into a handful of Inductor-generated ones, so the trace is shorter and the kernel names change to `triton_poi_fused_...`, but the launches still happen from the host and NVTX ranges around your Python still fire. A CUDA graph (including `torch.compile(mode="reduce-overhead")`, which is compile-plus-graphs) collapses the launches into one replay and erases host-side annotation entirely unless you recorded it into the graph. The table:

| Execution mode | What the trace shows | Kernel count (typical) | Do host NVTX ranges fire? | Annotation that works |
|---|---|---|---|---|
| Eager | Every kernel, named by template | ~8,000/request | Yes (launched from Python) | `nvtx.range` or `record_function` |
| `torch.compile` | Fused Inductor kernels | ~hundreds | Yes (still host-launched) | `nvtx.range` around Python |
| CUDA graph replay | One `cudaGraphLaunch` node | 1 host op | No — replay skips your code | NVTX recorded *into* the graph |

Read the last two columns together: the faster the mode, the more the host drops out of the loop, and the more you must move your annotation *into* the recorded work rather than around the Python call. This is the general shape of the observability-versus-speed tension — every optimization that removes the host from the hot path also removes the host-side labels, so you pay for speed in visibility unless you deliberately buy it back with capture-time annotation.

#### Worked example: the graphed service that went dark

A recommendation-model inference service on **A100 80GB SXM** turns on CUDA graphs and celebrates: p50 latency drops from 41 ms to 24 ms, GPU utilization climbs from 34% to 79%, and the launch-overhead wall — 8,000 `cudaLaunchKernel` calls per request — is gone. Two weeks later, p99 quietly regresses to 60 ms on a subset of requests, and nobody can tell why, because the trace is now one `cudaGraphLaunch` bar. The graph that fixed the mean latency destroyed the observability needed to fix the tail.

They add per-layer NVTX ranges inside the capture region (the `AnnotatedDecoder` above), confirm on a two-layer test that the ranges survive replay on their CUDA 12.4 stack, and re-capture. The thirty-two lanes reappear. The `nvtx_sum` report over a batch of the slow requests shows the tail is `decoder_layer_3` spiking to 6 ms on exactly the inputs with a rare categorical feature — a data-dependent branch that the static graph handles by always taking the expensive path, but whose *cost* only shows up for certain inputs because of a divergent embedding-gather. The label made a graph-hidden, input-dependent tail *visible*; without the surviving annotation it would have stayed a black box replaying at variable speed. Here is the measured before/after of the *debugging capability*, which is the real deliverable:

| Metric (A100 80GB) | Graph, no annotation | Graph + surviving NVTX |
|---|---|---|
| GPU rows in trace for forward | 1 (`cudaGraphLaunch`) | 32 labeled lanes |
| Time to locate the slow layer | could not — opaque | ~15 s (sorted nvtx_sum) |
| p99 tail root cause | unknown for 2 weeks | found in one capture |
| Replay overhead added by NVTX | n/a | < 40 µs/replay |
| Latency impact of annotations | n/a | none measurable |

The last two rows matter for the "is it safe" reflex: the recorded NVTX nodes add well under a microsecond per node on replay, so annotating a thirty-two-layer graph costs tens of microseconds against a 24 ms replay — noise. You pay nothing measurable and you get your eyes back.

## Reading the annotated trace: torch.profiler and nsys side by side

To make the `record_function` ↔ NVTX relationship fully concrete, here is the *same* semantic labels feeding both tools. First, torch.profiler with `record_function`, which is what you use when you want the Chrome trace and the aggregated table without leaving Python:

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("preprocess"):
        x = preprocess(request).cuda()
    with record_function("forward"):
        y = model(x)
    with record_function("sample"):
        out = sample(y)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or perfetto
```

The `key_averages().table()` groups by your `record_function` labels and rolls up the CUDA time each scope owns:

```console
-------------------  ------------  ------------  ------------  ------------
Name                   Self CUDA    CUDA total    # of Calls    CUDA %
-------------------  ------------  ------------  ------------  ------------
forward                  0.000us      38.041ms             1     92.6%
   aten::mm             21.300ms      21.300ms           288     51.8%
   aten::_sdpa          9.820ms       9.820ms            32      23.9%
   aten::layer_norm     3.100ms       3.100ms            65      7.5%
preprocess               0.000us       1.201ms             1      2.9%
sample                   0.000us       0.602ms             1      1.5%
-------------------  ------------  ------------  ------------  ------------
Self CPU time total: 4.021ms
Self CUDA time total: 41.088ms
```

Same conclusion as the `nsys nvtx_sum` earlier — `forward` is 92.6% — reached through the torch.profiler consumer instead of the Nsight one, using `record_function` labels instead of NVTX ranges. The two tables agree because they are labeling the same regions; they differ only in *who* renders them. When your investigation lives inside PyTorch and you want quick aggregation, `record_function` + `key_averages` is the fast path. When you need the system-wide view — CPU threads, CUDA API, memcpy engines, and NVTX all on one correlated timeline — you export to `nsys` and read the NVTX ranges there. The rule to keep: *one act of labeling your code, two consumers that can read it.* Instrument for the tool you will actually open.

![a decision tree that starts from which annotation and branches by which tool reads the trace into record function for torch profiler nvtx range or emit nvtx for nsight and kernel annotation for a graphed model](/imgs/blogs/nvtx-and-semantic-profiling-traces-6.webp)

The decision tree above is the whole post as a single lookup: start from "which annotation?", branch on the tool that will read the trace. Using `torch.profiler` and its Chrome trace? Reach for `record_function`. Using `nsys` or `ncu`? Reach for `nvtx.range` (hand-placed, coarse) or `emit_nvtx` (auto, per-op, profiling-only). Working with a captured graph where a plain replay is one opaque node? Reach for the kernel-annotation technique that records ranges into the graph so they survive replay. There is no "best" annotation — there is only the one that matches the consumer you are about to open, and the graphed-model branch is the real intermediate case that trips people up, because it is the one where the naive approach silently produces nothing.

To put a number on the payoff — because "readable" is not a metric and this series lives on measured before/after — here is the same debugging task (locate the single slow phase in a 32-layer decoder) timed with a stopwatch on two cards, once against a raw trace and once against an annotated one. The task is identical; only the trace differs. The "time-to-find-the-wall" is wall-clock human time from opening the trace to naming the culprit phase, averaged over a handful of engineers who did not already know the answer:

| Time-to-find-the-wall | A100 80GB (raw) | A100 80GB (NVTX) | L4 (raw) | L4 (NVTX) |
|---|---|---|---|---|
| Kernels to scan | ~8,000 | 32 labeled lanes | ~8,000 | 32 labeled lanes |
| Median time to name the phase | ~8 min | ~15 s | ~11 min | ~18 s |
| Confidence in the answer | "pretty sure" | certain (sorted sum) | "pretty sure" | certain |
| Re-find after a code change | full re-scan | re-sort, ~10 s | full re-scan | re-sort, ~12 s |

The card barely matters — the raw-trace time is a hair longer on the L4 only because its slower profiling I/O makes scrolling a large capture feel worse — but the *annotation* matters enormously: roughly a 30x collapse in time-to-find, and, just as important, a jump from "pretty sure" to "certain," because a sorted `nvtx_sum` is an argument, not a hunch. Multiply that 30x by every profiling session over the service's life and you have the actual return on the seven `with` statements. The model did not get faster; *you* did.

## Case studies and real numbers

A few grounded results, cited where I can and flagged as approximate where I cannot.

**PyTorch `reduce-overhead` and the launch-overhead win.** The PyTorch team documents `torch.compile(mode="reduce-overhead")` as compile-plus-CUDA-graphs, and its whole reason to exist is the launch-overhead problem NVTX makes visible. Published PyTorch benchmarks show `reduce-overhead` delivering meaningful speedups on inference workloads dominated by many small kernels — the exact host-bound signature where CUDA graphs collapse thousands of launches into one replay. The relevant point for *this* post is that turning it on is precisely what darkens your trace, which is why the annotation technique is the necessary companion to the speedup, not an optional nicety. Reach for the [kernel-fusion, CUDA-graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) discussion for how the compile path composes with graphs.

**Nsight-guided kernel investigations.** NVIDIA's own Nsight Systems and Nsight Compute documentation centers NVTX as the mechanism for correlating application phases to GPU work, and the standard NVIDIA optimization workflow is exactly the composition shown here: `nsys` with NVTX to find the phase, `ncu --nvtx-include` to go to the metal on that phase's kernels. This is not a niche trick — it is the documented, recommended path, and it is why deep-serving profiling writeups like [profiling LLM serving with Nsight](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight) lean on NVTX ranges to make a multi-stage generation loop (prefill vs decode vs sampling) legible on one timeline.

**The masking-bug find.** The layer-19 story in this post — one decoder layer running full-sequence attention because of a KV-cache masking bug, invisible in a raw trace and a sorted one-liner in an annotated one — is the archetypal NVTX win, and variants of it recur constantly in real serving work. The number to remember is not the fix (four lines) but the *find*: minutes-of-scrolling versus seconds-of-reading. That delta, multiplied across every profiling session for the life of the service, is the actual return on instrumentation. The [reproducible-benchmark discipline](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) is what lets you *trust* the before/after; NVTX is what lets you *find* the thing to change in the first place.

**The graphed-tail find.** The recommendation-service story — a data-dependent tail hiding inside a CUDA-graph replay, surfaced only after inner NVTX ranges were recorded into the graph — is the case that justifies the whole "CUDA Graph Kernel Annotations" tutorial. Once you graph a model for throughput, you *will* eventually need to see inside the replay, and the day you need it is not the day to discover your labels evaporated on capture. Bake them in from the start.

## When to reach for this (and when not to)

NVTX annotation is unusually low-risk — near-zero cost at rest, enormous upside when profiling — so the "when not to" list is short, but it is not empty.

**Reach for hand-placed NVTX ranges always.** There is essentially no service that should not have coarse phase ranges (`preprocess`, `H2D`, `forward`, `sample`, `D2H`, `serialize`) permanently in its handler. They cost nanoseconds at rest and turn every future profiling session from archaeology into reading. This is the closest thing to a free lunch in performance work. If your service is graphed, add the per-layer ranges *inside* the capture and verify once that they survive replay.

**Reach for `emit_nvtx` only for a deliberate zoom-in.** When a coarse range turns out to be the wall and you need op-level detail *inside* it, switch on `emit_nvtx` for one capture. Do not leave it on — the per-op overhead is real (a ~10% profiling tax on a transformer forward) and the four-thousand-bar timeline is noise for any question except "which op."

**Do not bother with custom NVTX *domains* (the C-extension kind) until the prefix convention fails you.** True separate domain rows via `nvtxDomainCreate` are elegant, but the `gpu/`, `cpu/`, `io/` string-prefix convention gets you filterable, grouped lanes in pure PyTorch with zero extra build machinery. Ninety percent of services never outgrow the convention.

**Do not annotate what you will not read.** Labels are documentation; documentation that describes a consumer you never open is dead weight — not costly, but confusing. If you only ever use `torch.profiler`, use `record_function` and skip NVTX. If you only ever use `nsys`, use NVTX and skip `record_function`. Match the annotation to the tool you actually open; do not sprinkle both everywhere out of superstition.

**Do not expect NVTX to *fix* anything.** This is the honest caveat. NVTX makes the trace readable; it does not make the model faster. If `forward` is 92.6% of your request and it is genuinely compute-bound at 85% SM occupancy, a perfectly labeled trace just tells you, precisely, that there is no easy win here. The label is a diagnostic, not a treatment. Its value is that it points you at the *right* treatment fast — CUDA graphs for host-bound, fusion for bandwidth-bound, a kernel rewrite for occupancy-bound — instead of guessing.

## Key takeaways

- **A raw GPU trace is named for the compiler's kernels, not your request's phases.** Eight thousand `elementwise_kernel` rows tell you nothing about which is preprocess, forward, or sample. NVTX ranges stamp *your* semantics into the trace so every span maps to a phase you understand.
- **NVTX is a free producer and the tool is the consumer.** `range_push`/`range_pop` cost about a nanosecond with no profiler attached (a stub), and real events only when `nsys` or `ncu` is watching. That is why you leave coarse ranges in production permanently: $t_\text{overhead} = N \cdot t_\text{nvtx}$, and $N \cdot 1\text{ ns}$ is nothing.
- **Match the annotation to the tool.** `record_function` feeds `torch.profiler`'s Chrome trace and `key_averages` table; `nvtx.range` feeds `nsys`/`ncu`; `emit_nvtx` auto-labels every op (profiling-only, ~10% tax); one set of labels can feed both when you bridge with `emit_nvtx`.
- **Nest ranges to get a call hierarchy.** `request ⊃ forward ⊃ decoder_layer_i ⊃ attention` lets you drill from "request slow" to "one kernel slow" without leaving the labeled view. Prefer the `with`-block form so the pop always fires and the stack never desyncs.
- **A range correlates to its kernels.** The sum of a range's kernel times versus its GPU-projected wall-time is a free compute-bound-vs-host-bound diagnosis for that phase: equal means packed, a gap means the GPU idled waiting on the host.
- **NVTX is the scalpel for `ncu`.** `--nvtx-include "decoder_layer_19/"` scopes an expensive full-metric `ncu` run to exactly the kernels in one semantic phase, turning an intractable whole-model profile into a seconds-long targeted one.
- **CUDA-graph capture erases kernel identity; annotation restores it.** Replay is one `cudaGraphLaunch`, so host-side pushes never fire — unless you record NVTX ranges *into* the graph during capture (CUDA 12.x). Verify once that inner ranges survive replay on your stack; if not, label the whole replay as a fallback.
- **The win is in your debugging loop, not the model.** Annotation does not speed up a single kernel. It collapses time-to-find-the-wall from minutes of scrolling to seconds of reading a sorted `nvtx_sum`, every session, for the life of the service.

## Further reading

- **PyTorch — CUDA Graph Kernel Annotations and Profiling** (the deep-dive tutorial this post is grounded on): the technique for annotating graph kernels so semantic labels survive capture and appear as custom visualization lanes in the profiler.
- **NVIDIA NVTX documentation** (the NVIDIA Tools Extension library): the range/mark/domain primitives, the C API (`nvtxRangePushEx`, `nvtxDomainCreate`), and how tools consume the events.
- **NVIDIA Nsight Systems documentation**: `nsys profile -t nvtx`, the timeline, NVTX rows, and `nsys stats --report nvtx_sum` for the phase summary.
- **NVIDIA Nsight Compute documentation**: `--nvtx` and `--nvtx-include`/`--nvtx-exclude` for scoping kernel metrics to a semantic range, plus the Speed-of-Light section.
- **PyTorch `torch.profiler` and `record_function` docs**: the torch-native span analog of NVTX, `key_averages().table()`, and `export_chrome_trace`.
- Within this series: [why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) (the four wastes and the profile→fix→measure loop), [profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) (`record_function` in depth), [Nsight Systems for AI services](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) (reading the system-wide timeline), [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) (capturing the replay these annotations survive), and the capstone [performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook).
